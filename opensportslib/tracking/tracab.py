"""Indexed, dependency-free reader for calibrated TRACAB text DAT streams.

The input descriptor owns timing and team mapping. A referenced metadata JSON
supplies rosters and geometry; UTC is never inferred from source frame counters.
"""
from __future__ import annotations

import bisect
import copy
import io
import hashlib
import json
import math
import os
import threading
from datetime import datetime, timedelta, timezone

from .native import NativeDocument, TracabDatReader, parse_dat_record, read_document, metadata_conflicts

PARSER_VERSION = 4


def parse_utc(value):
    """Return an aware UTC datetime; naive OSL timestamps mean UTC."""
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def utc_text(value):
    return value.astimezone(timezone.utc).replace(tzinfo=None).isoformat(" ", timespec="microseconds")


def _number(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value, name):
    result = _number(value, name)
    if result != int(result):
        raise ValueError(f"{name} must be an integer")
    return int(result)


def load_tracab_documents(descriptor, *, base_dir=None, metadata=None, xml_metadata=None):
    """Read referenced source documents without interpreting unknown fields."""
    documents = {}
    for key, format in (("metadata_path", "json"), ("tf05_path", "xml"), ("tf08_path", "xml")):
        reference = descriptor.get(key)
        supplied = metadata if key == "metadata_path" else (xml_metadata or {}).get(key)
        if not reference and supplied is None:
            continue
        try:
            if isinstance(supplied, NativeDocument):
                document = supplied
            elif supplied is not None:
                raw = json.dumps(supplied).encode() if format == "json" else supplied
                document = read_document(io.BytesIO(raw), format=format)
                document.source = str(reference or key)
            else:
                path = os.fspath(reference)
                if not os.path.isabs(path):
                    if base_dir is None:
                        raise ValueError("relative companion path requires metadata_base_dir (the manifest/media root)")
                    path = os.path.join(base_dir, path)
                document = read_document(path, format=format)
            if format == "xml" and document.feed_id != key[:4].upper():
                raise ValueError(f"expected {key[:4].upper()} feed, observed {document.feed_id!r}; feeds are not interchangeable")
            documents[key] = document
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"TRACAB metadata {reference}: {exc}") from exc
    return documents


def resolve_tracab_descriptor(descriptor, *, base_dir=None, metadata=None, xml_metadata=None, documents=None):
    """Resolve source metadata into an internal view, leaving the manifest intact.

    Calibrated periods reference JSON PhaseN bounds by ``period``. Only UTC
    anchors and mappings absent from the delivery need to live in the manifest.
    """
    result = copy.deepcopy(descriptor)
    documents = documents if documents is not None else load_tracab_documents(
        result, base_dir=base_dir, metadata=metadata, xml_metadata=xml_metadata)
    if "metadata_path" not in documents:
        return result
    document = documents["metadata_path"]
    data = document.value
    try:
        conflicts = metadata_conflicts(document, [doc for doc in documents.values() if doc.format == "xml"])
        if conflicts:
            raise ValueError("; ".join(conflicts))
        fps = _number(data["FrameRate"], "FrameRate")
        derived = {
            "pitch_length_m": _number(data["PitchLongSide"], "PitchLongSide") / 100,
            "pitch_width_m": _number(data["PitchShortSide"], "PitchShortSide") / 100,
            "rosters": {
                side: [{"jerseyNum": _integer(player["JerseyNo"], "JerseyNo"),
                        "playerId": player["PlayerID"], "teamId": data[title + "Team"].get("TeamID"),
                        "start_frame": player.get("StartFrameCount"), "end_frame": player.get("EndFrameCount")}
                       for player in data[title + "Team"]["Players"]]
                for side, title in (("home", "Home"), ("away", "Away"))
            },
        }
        if "fps" in result and result["fps"] != fps:
            raise ValueError("fps disagrees with metadata FrameRate")
        result["fps"] = fps
        settings = result.setdefault("metadata", {})
        for key, value in derived.items():
            if key in settings and settings[key] != value:
                raise ValueError(f"metadata.{key} disagrees with {document.source}; remove the duplicate")
            settings[key] = value
        for period in settings.get("periods", []):
            if "period" not in period:
                raise ValueError("metadata.periods must reference a source JSON phase using 'period'; remove duplicated frame bounds")
            number = _integer(period["period"], "period")
            for bound in ("start", "end"):
                source_key = f"Phase{number}{bound.title()}Frame"
                value = _integer(data[source_key], source_key)
                key = bound + "_frame"
                if key in period and period[key] != value:
                    raise ValueError(f"{key} conflicts with {document.source}:{source_key}")
                period[key] = value
            if period["start_frame"] == period["end_frame"] == 0:
                raise ValueError(f"Phase{number} has zero/zero bounds; provide calibration only for populated periods")
        result["_source_fingerprints"] = {key: hashlib.sha256(doc.raw).hexdigest() for key, doc in documents.items()}
        return result
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"TRACAB metadata {document.source}: {exc}") from exc


class TracabTrackingReader:
    """Explicit normalized view of classic DAT: centered meters and video time.

    Use TracabDatReader for default native access without transformations.

    ``source`` may be a filename or binary seekable stream. The reader owns the
    stream until close(). Index entries include virtual empty rows delimiting
    gaps, so nearest-frame playback cannot leak a future playing frame into a
    break. ``row_at`` uses frame validity intervals, not nearest-neighbour fill.
    """

    def __init__(self, source, descriptor, progress_callback=None, *, metadata_base_dir=None, metadata=None, xml_metadata=None):
        self.descriptor = copy.deepcopy(descriptor)
        self.path = os.fspath(source) if isinstance(source, (str, os.PathLike)) else None
        self.name = self.path or str(descriptor.get("path", "<stream>"))
        self._stream = None if self.path else source
        self._pid = os.getpid()
        self._lock = threading.RLock()
        try:
            self.documents = load_tracab_documents(descriptor, base_dir=metadata_base_dir, metadata=metadata, xml_metadata=xml_metadata)
            self.descriptor = resolve_tracab_descriptor(descriptor, documents=self.documents)
            self.transformations = {
                "player_xy": "source / 100 (classic centimeters to meters)",
                "ball_xy": "source / 100 (classic centimeters to meters)",
                "ball_z": "source / 100 (classic height centimeters to meters)",
                "pitch_dimensions": "source / 100 (centimeters to meters)",
                "speed": "unchanged; no unit assumption", "axes": "unchanged",
                "timing": "explicit per-period UTC anchors and optional calibrated frame intervals",
            }
            self._configure()
            self._index(progress_callback)
        except Exception as exc:
            self.close()
            raise ValueError(f"{self.name}: {exc}") from exc

    def _configure(self):
        d = self.descriptor
        self.fps = _number(d["fps"], "fps")
        self.duration_ms = _integer(d["duration_ms"], "duration_ms")
        self.origin_utc = parse_utc(d["UTC_time_start"])
        if not 0 < self.fps <= 1000 or self.duration_ms <= 0:
            raise ValueError("fps must be in (0, 1000] and duration_ms positive")
        self.interval_ms = 1000 / self.fps
        metadata = d["metadata"]
        self.team_codes = {side: _integer(metadata["team_codes"][side], "team code") for side in ("home", "away")}
        if len(set(self.team_codes.values())) != 2:
            raise ValueError("home and away team codes must differ")
        self.rosters = {}
        for side, players in metadata.get("rosters", {}).items():
            roster = {}
            for player in players:
                jersey = _integer(player["jerseyNum"], "jerseyNum")
                if jersey <= 0:
                    raise ValueError(f"Duplicate/invalid {side} jersey {jersey}")
                existing = roster.setdefault(jersey, [])
                start, end = player.get("start_frame"), player.get("end_frame")
                for other in existing:
                    left, right = other.get("start_frame"), other.get("end_frame")
                    if None in (start, end, left, right) or not (end < left or start > right):
                        raise ValueError(f"Duplicate/invalid or overlapping {side} jersey {jersey}; identity is ambiguous")
                existing.append(dict(player))
            self.rosters[side] = roster
        self.pitch_length_m = _number(metadata["pitch_length_m"], "pitch_length_m")
        self.pitch_width_m = _number(metadata["pitch_width_m"], "pitch_width_m")
        if min(self.pitch_length_m, self.pitch_width_m) <= 0:
            raise ValueError("Pitch dimensions must be positive")
        self.periods = []
        for period in metadata["periods"]:
            start = _integer(period["start_frame"], "start_frame")
            end = _integer(period["end_frame"], "end_frame")
            offset = (parse_utc(period["start_utc"]) - self.origin_utc).total_seconds() * 1000
            interval = _number(period.get("video_ms_per_source_frame", self.interval_ms), "video_ms_per_source_frame")
            if interval < 1:
                raise ValueError("video_ms_per_source_frame must be at least 1 ms")
            last = offset + (end - start) * interval
            if start < 0 or end < start or offset < 0 or last >= self.duration_ms:
                raise ValueError("Period frames/timestamps outside declared bounds; correct calibration")
            if self.periods:
                prev_start, prev_end, prev_offset, prev_interval = self.periods[-1]
                if start <= prev_end or offset < prev_offset + (prev_end - prev_start + 1) * prev_interval:
                    raise ValueError("Periods overlap or are non-monotonic; correct calibration")
            self.periods.append((start, end, offset, interval))
        if not self.periods:
            raise ValueError("metadata.periods must contain calibrated playing periods")

    def _handle(self):
        if self.path and (self._stream is None or self._pid != os.getpid()):
            self.close()
            self._stream = open(self.path, "rb")
            self._pid = os.getpid()
        return self._stream

    def _decode(self, line, line_number):
        record = parse_dat_record(line, source=self.name, line_number=line_number)
        frame = int(record.frame_id)
        payload = {"homePlayers": [], "awayPlayers": [], "balls": []}
        seen = set()
        sides = {code: side for side, code in self.team_codes.items()}
        for target in record.targets:
            values = target.fields
            code, jersey = int(values[0]), int(values[2])
            side = sides.get(code)
            if side is None or jersey <= 0:
                continue  # Native access retains every official/unidentified target.
            if (side, jersey) in seen:
                raise ValueError(f"line {line_number}: Duplicate {side} jersey {jersey} in frame {frame}")
            seen.add((side, jersey))
            roster = self.rosters.get(side)
            candidates = roster.get(jersey, []) if roster is not None else []
            if roster is not None and not candidates:
                raise ValueError(f"line {line_number}: Unknown {side} jersey {jersey}; check roster/team mapping")
            active = [p for p in candidates if p.get("start_frame") is None or p.get("end_frame") is None
                      or p["start_frame"] <= frame <= p["end_frame"]]
            player = {key: value for key, value in (active[0] if len(active) == 1 else {}).items()
                      if key not in {"start_frame", "end_frame"}}
            player.update(x=float(values[3]) / 100, y=float(values[4]) / 100,
                          jerseyNum=jersey, trackId=values[1], teamCode=values[0], shirtNumber=values[2],
                          speed=float(values[5]))
            if candidates and not active:
                player["identityStatus"] = "unresolved_outside_roster_interval"
            if target.additional_fields:
                player["additional_fields"] = list(target.additional_fields)
            payload[side + "Players"].append(player)
        for ball in record.balls:
            values = ball.fields
            payload["balls"].append(dict(x=float(values[0]) / 100, y=float(values[1]) / 100,
                                        z=float(values[2]) / 100, speed=float(values[3]),
                                        possession=values[4], state=values[5], flags=list(ball.additional_fields)))
        return frame, payload

    def native_frame(self, index):
        """Unmodified source frame behind a playback row, or None for a gap."""
        entry = self.entries[index]
        if entry is None:
            return None
        with self._lock:
            stream = self._handle()
            stream.seek(entry[0])
            return parse_dat_record(stream.readline(), source=self.name, line_number=entry[1])

    def _index(self, progress):
        stream = self._handle()
        stream.seek(0, 2)
        total = stream.tell()
        stream.seek(0)
        self.entries = []
        self.time_axis_ms = []
        self.source_frame_count = 0
        self.playing_frame_count = 0
        boundaries = {v for start, end, _, _ in self.periods for v in (start, end)}
        previous_frame = -1
        active_end = None

        def append(time, entry):
            time = round(time)
            if self.time_axis_ms and time <= self.time_axis_ms[-1]:
                if entry is None and time == self.time_axis_ms[-1]:
                    return
                raise ValueError("Non-monotonic mapped timestamps; correct calibration/fps")
            self.time_axis_ms.append(time)
            self.entries.append(entry)

        append(0, None)
        line_number = 0
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            line_number += 1
            if not line.strip():
                continue
            frame, _ = self._decode(line, line_number)
            if frame <= previous_frame:
                raise ValueError(f"line {line_number}: non-monotonic source frame {frame}")
            previous_frame = frame
            self.source_frame_count += 1
            period = next((p for p in self.periods if p[0] <= frame <= p[1]), None)
            if period is not None:
                time = round(period[2] + (frame - period[0]) * period[3])
                if time >= self.duration_ms:
                    raise ValueError(f"line {line_number}: mapped timestamp outside duration")
                if active_end is not None and time > active_end:
                    append(active_end, None)
                if self.entries[-1] is None and time > self.time_axis_ms[-1] + 1:
                    append(time - 1, None)
                if time == 0 and self.entries == [None]:
                    self.entries.clear()
                    self.time_axis_ms.clear()
                append(time, (offset, line_number, frame))
                active_end = min(self.duration_ms, round(period[2] + (frame - period[0] + 1) * period[3]))
                self.playing_frame_count += 1
                boundaries.discard(frame)
            if progress and line_number % 2000 == 0:
                progress(stream.tell(), total, "Indexing TRACAB frames…")
        if boundaries or not self.playing_frame_count:
            raise ValueError(f"Missing calibrated period boundaries: {sorted(boundaries)}")
        if active_end < self.duration_ms:
            append(active_end, None)
        if self.entries[-1] is None and self.time_axis_ms[-1] < self.duration_ms - 1:
            append(self.duration_ms - 1, None)
        if progress:
            progress(total, total, "Indexing TRACAB frames…")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        payload = {"homePlayers": [], "awayPlayers": [], "balls": []}
        if entry is not None:
            with self._lock:
                stream = self._handle()
                stream.seek(entry[0])
                _, payload = self._decode(stream.readline(), entry[1])
        time = self.time_axis_ms[index]
        return {"videoTimeMs": time, "sourceFrame": entry[2] if entry else None,
                "timestamp_utc": utc_text(self.origin_utc + timedelta(milliseconds=time)), **payload}

    def row_at(self, position_ms):
        index = bisect.bisect_right(self.time_axis_ms, position_ms) - 1
        if index >= 0 and position_ms < self.duration_ms:
            row = self[index]
            if row["sourceFrame"] is None:
                return row
            frame = row["sourceFrame"]
            period = next(p for p in self.periods if p[0] <= frame <= p[1])
            frame_end = round(period[2] + (frame - period[0] + 1) * period[3])
            if position_ms < frame_end:
                return row
        return {"videoTimeMs": position_ms, "sourceFrame": None, "homePlayers": [], "awayPlayers": [], "balls": []}

    def iter_uniform_rows(self):
        """Native-rate video timeline, retaining empty time rather than joining halves."""
        for index in range(math.ceil(self.duration_ms * self.fps / 1000)):
            time = round(index * self.interval_ms, 6)
            yield {**self.row_at(time), "videoTimeMs": time}

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        if not self.path:
            with self._lock:
                stream = self._handle()
                position = stream.tell()
                stream.seek(0)
                state["_stream_bytes"] = stream.read()
                stream.seek(position)
        state["_stream"] = None
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        content = state.pop("_stream_bytes", None)
        self.__dict__.update(state)
        self._lock = threading.RLock()
        self._stream = io.BytesIO(content) if content is not None else None

"""Lossless access to the observed classic ASCII TRACAB delivery variant.

DAT has positional fields, not field names. The names below describe that
classic layout; all field tokens and original bytes remain available. Nothing
in this module converts units, infers UTC, drops targets, or decodes heatmaps.
"""
from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

TARGET_FIELDS = ('team_code', 'tracking_id', 'shirt_number', 'x', 'y', 'speed')
BALL_FIELDS = ('x', 'y', 'z', 'speed', 'possession', 'status')


@dataclass(frozen=True)
class Diagnostic:
    source: str
    line: int | None
    frame: str | None
    code: str
    message: str

    def __str__(self):
        return f'{self.source}: line {self.line}, frame {self.frame}: {self.code}: {self.message}'


@dataclass(frozen=True)
class NativeTarget:
    fields: tuple[str, ...]

    @property
    def values(self):
        return dict(zip(TARGET_FIELDS, self.fields))

    @property
    def additional_fields(self):
        return self.fields[len(TARGET_FIELDS):]


@dataclass(frozen=True)
class NativeBall:
    fields: tuple[str, ...]

    @property
    def values(self):
        return dict(zip(BALL_FIELDS, self.fields))

    @property
    def additional_fields(self):
        return self.fields[len(BALL_FIELDS):]


@dataclass
class NativeFrame:
    raw: bytes
    source: str
    line_number: int
    frame_id: str | None = None
    targets: list[NativeTarget] = field(default_factory=list)
    balls: list[NativeBall] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    rejected: bool = False


def parse_dat_record(raw, *, source='<stream>', line_number=1, mode='strict'):
    """Parse one line, retaining every byte/token, including rejected records."""
    if mode not in {'strict', 'permissive'}:
        raise ValueError("mode must be 'strict' or 'permissive'")
    record = NativeFrame(raw=raw, source=source, line_number=line_number)
    try:
        blocks = raw.decode('utf-8-sig').rstrip('\r\n').split(':')
        record.frame_id = blocks[0] if blocks else None
        if len(blocks) not in (3, 4) or (len(blocks) == 4 and blocks[-1] != ''):
            raise ValueError('unsupported DAT variant: expected frame:targets:balls:')
        int(record.frame_id)
        for block, cls, destination, name in (
            (blocks[1], NativeTarget, record.targets, 'target'),
            (blocks[2], NativeBall, record.balls, 'ball'),
        ):
            items = block.split(';')
            for index, item in enumerate(items):
                if item == '':
                    # Empty block/trailing separators are valid and stay in raw.
                    if any(items[index:]):
                        record.diagnostics.append(Diagnostic(source, line_number, record.frame_id,
                                                             'empty_target_slot', f'Empty {name} slot {index} retained in raw bytes'))
                    continue
                fields = tuple(item.split(','))
                destination.append(cls(fields))
                if len(fields) < 6:
                    raise ValueError(f'Malformed {name} record: expected at least 6 fields; got {len(fields)}')
                if name == 'target':
                    for token in fields[:3]:
                        int(token)
                    numeric = fields[3:6]
                else:
                    numeric = fields[:4]
                for token in numeric:
                    if not math.isfinite(float(token)):
                        raise ValueError(f'{name} coordinate/speed must be finite')
                if len(fields) > 6:
                    record.diagnostics.append(Diagnostic(source, line_number, record.frame_id,
                                                         'additional_fields', f'{name}: {len(fields) - 6} extra fields retained without interpretation'))
    except (ValueError, UnicodeError) as exc:
        record.rejected = True
        diagnostic = Diagnostic(source, line_number, record.frame_id, 'malformed_record', str(exc))
        record.diagnostics.append(diagnostic)
        if mode == 'strict':
            raise ValueError(str(diagnostic)) from exc
    return record


@dataclass
class NativeDocument:
    source: str
    raw: bytes
    format: str
    value: object
    namespaces: list[tuple[str, str]] = field(default_factory=list)

    @property
    def feed_id(self):
        return self.value.get('sTracabFeedId') if self.format == 'xml' else None

    def heatmaps(self):
        """All heatmap attributes, with element paths and exact string values."""
        if self.format != 'xml':
            return
        def walk(node, path):
            for key, value in node.attrib.items():
                if 'heatmap' in key.lower():
                    yield {'source': self.source, 'element': path, 'attribute': key, 'encoded': value}
            for index, child in enumerate(node):
                yield from walk(child, f'{path}/{child.tag}[{index}]')
        yield from walk(self.value, f'/{self.value.tag}')


def read_document(source, *, format):
    """Preserve raw JSON/XML and parsed unknown fields, namespaces and children."""
    if isinstance(source, (str, os.PathLike)):
        name = os.fspath(source)
        with open(name, 'rb') as stream:
            raw = stream.read()
    else:
        name = str(getattr(source, 'name', '<stream>'))
        raw = source.read()
    try:
        if format == 'json':
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError('expected a JSON object')
            return NativeDocument(name, raw, format, value)
        if format == 'xml':
            namespaces = [item for _, item in ET.iterparse(io.BytesIO(raw), events=('start-ns',))]
            parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True))
            return NativeDocument(name, raw, format, ET.fromstring(raw, parser=parser), namespaces)
        raise ValueError(f'Unsupported document format {format!r}')
    except (ValueError, ET.ParseError) as exc:
        raise ValueError(f'{name}: {exc}') from exc


def metadata_conflicts(json_document, xml_documents):
    """Check only observed identity fields; do not guess unknown XML semantics."""
    conflicts = []
    data = json_document.value
    for xml in xml_documents:
        for node in xml.value.iter():
            tag = str(node.tag).rsplit('}', 1)[-1]
            if tag == 'TracabDocument':
                for json_key, xml_key in (('GameID', 'iMatchId'), ('CompetitionID', 'iCompetitionId'), ('SeasonID', 'iSeasonId')):
                    if json_key in data and xml_key in node.attrib and str(data[json_key]) != node.get(xml_key):
                        conflicts.append(f'{json_document.source}:{json_key}={data[json_key]!r} conflicts with {xml.source}:{xml_key}={node.get(xml_key)!r}')
                for side in ('Home', 'Away'):
                    team = data.get(side + 'Team', {})
                    key = 'i' + side + 'TeamId'
                    if 'TeamID' in team and key in node.attrib and str(team['TeamID']) != node.get(key):
                        conflicts.append(f'{json_document.source}:{side}Team.TeamID={team["TeamID"]!r} conflicts with {xml.source}:{key}={node.get(key)!r}')
            if tag in ('HomeTeam', 'AwayTeam'):
                team = data.get(tag, {})
                if 'TeamID' in team and 'iTeamID' in node.attrib and str(team['TeamID']) != node.get('iTeamID'):
                    conflicts.append(f'{json_document.source}:{tag}.TeamID conflicts with {xml.source}:{tag}.iTeamID')
                players = {str(p['PlayerID']): p for p in team.get('Players', [])}
                for player in node:
                    if str(player.tag).rsplit('}', 1)[-1] != 'Player':
                        continue
                    identity = player.get('iPlayerId')
                    if identity is None:
                        continue
                    if identity not in players:
                        conflicts.append(f'{xml.source}:{tag}.Player {identity!r} is unresolved in {json_document.source}')
                    elif 'iJersey' in player.attrib and str(players[identity].get('JerseyNo')) != player.get('iJersey'):
                        conflicts.append(f'{xml.source}:{tag}.Player {identity!r} jersey conflicts with {json_document.source}')
    return conflicts


class TracabDatReader:
    """Streaming native DAT reader. Iteration yields original records in file order.

    Strict mode rejects malformed required fields. Permissive mode yields the
    rejected record with its bytes and diagnostics. Timing anomalies are always
    reported and preserved. Supplied file objects remain owned by the caller.
    """
    def __init__(self, source, *, mode='strict'):
        if mode not in {'strict', 'permissive'}:
            raise ValueError("mode must be 'strict' or 'permissive'")
        self.source = source
        self.mode = mode

    def __iter__(self):
        owned = isinstance(self.source, (str, os.PathLike))
        stream = open(self.source, 'rb') if owned else self.source
        name = os.fspath(self.source) if owned else str(getattr(stream, 'name', '<stream>'))
        previous = None
        seen = set()
        try:
            for line_number, raw in enumerate(stream, 1):
                record = parse_dat_record(raw, source=name, line_number=line_number, mode=self.mode)
                if not record.rejected:
                    frame = int(record.frame_id)
                    if frame in seen:
                        record.diagnostics.append(Diagnostic(name, line_number, record.frame_id, 'duplicate_frame', 'Duplicate frame retained'))
                    if previous is not None:
                        if frame < previous:
                            record.diagnostics.append(Diagnostic(name, line_number, record.frame_id, 'out_of_order', 'Out-of-order frame retained'))
                        elif frame > previous + 1:
                            record.diagnostics.append(Diagnostic(name, line_number, record.frame_id, 'frame_gap', f'{frame - previous - 1} missing frame IDs; no rows inserted'))
                    seen.add(frame)
                    previous = frame
                yield record
        finally:
            if owned:
                stream.close()


def decode_heatmap(encoded, *, rows, columns, order, orientation, value_semantics):
    """Explicit grid view only; callers must supply an established specification.

    Digits remain strings. No seconds/probabilities conversion is performed.
    Orientation and value semantics are retained as supplied provenance.
    """
    if not orientation or not value_semantics or order not in {'row-major', 'column-major'}:
        raise ValueError('Heatmap decoding requires known dimensions, cell order, orientation and value semantics')
    if rows <= 0 or columns <= 0 or len(encoded) != rows * columns or any(c not in '0123456789' for c in encoded):
        raise ValueError('Heatmap dimensions/encoding do not match the supplied specification')
    grid = [[encoded[r * columns + c] if order == 'row-major' else encoded[c * rows + r]
             for c in range(columns)] for r in range(rows)]
    return {'encoded': encoded, 'grid': grid, 'order': order, 'orientation': orientation, 'value_semantics': value_semantics}

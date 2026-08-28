"""Event extraction and tracking-frame flattening for SN-GAR action spotting.

Defines the SN-GAR event set: the mapping from the source event stream to the
ten action labels, the priority resolution that gives one label per instant,
and the alignment filter that discards events the tracking clock cannot
localise.

The label rules in extract_expanded_annotations are the dataset contract.
Changing one changes the dataset, and results measured on earlier builds stop
being comparable. build_sngar_spotting.py calls these functions without
modifying them.

Reference totals over the 64-game corpus:

    94,285 extracted
    -4,963 resolved to one label per instant
    -1,383 dropped as unalignable
    87,939 final: train 62,159 / valid 12,091 / test 13,689
"""

import os
import bz2
import json
from collections import defaultdict

import pandas as pd


SPLITS = {
    "train": (0, 45),
    "valid": (45, 54),
    "test": (54, 64),
}


LABELS = [
    "PASS", "HEADER", "HIGH PASS", "OUT", "CROSS",
    "THROW IN", "SHOT", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL",
]


LABEL_PRIORITY = {
    "PASS": 0,
    "HIGH PASS": 1,
    "OUT": 2,
    "CROSS": 3,
    "SHOT": 4,
    "HEADER": 5,
    "PLAYER SUCCESSFUL TACKLE": 6,
    "THROW IN": 7,
    "FREE KICK": 8,
    "GOAL": 9,
}


POSITION_GROUPS = {
    "GK": "GK",
    "LCB": "DEF", "RCB": "DEF", "MCB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
    "CM": "MID", "AM": "MID", "DM": "MID", "LM": "MID", "RM": "MID",
    "CF": "FWD", "LW": "FWD", "RW": "FWD",
}


GAME_EVENT_KEYS = ["game_event_type", "player_name", "player_id", "team_id", "home_team", "video_url"]


def build_position_mapping(jsonl_path):
    position_map = defaultdict(lambda: defaultdict(lambda: None))
    # Home/away team ids are static for a game. Resolving them once here lets
    # positions be applied to every frame's players rather than only the
    # sparse frames that carry a game_event.
    home_team_id = None
    away_team_id = None

    with bz2.open(jsonl_path, "rt") as f:
        for line in f:
            try:
                frame = json.loads(line)
                game_event = frame.get("game_event")

                if game_event and isinstance(game_event, dict):
                    team_id = game_event.get("team_id")
                    shirt_num = game_event.get("shirt_number")
                    position = game_event.get("position_group_type")

                    if team_id and shirt_num and position:
                        position_map[str(team_id)][str(shirt_num)] = position

                    if team_id:
                        if game_event.get("home_team"):
                            home_team_id = team_id
                        else:
                            away_team_id = team_id
            except Exception:
                continue

    return dict(position_map), home_team_id, away_team_id


def add_positions_to_players(players_list, team_id, position_map):
    if not players_list or not isinstance(players_list, list):
        return players_list

    team_positions = position_map.get(str(team_id), {})

    for player in players_list:
        if isinstance(player, dict):
            jersey = str(player.get("jerseyNum", ""))
            position = team_positions.get(jersey)
            player["position"] = position if position else None
            player["positionGroup"] = POSITION_GROUPS.get(position) if position else None

    return players_list


def flatten_frame(frame, position_map, home_team_id=None, away_team_id=None):
    flat = {
        "videoTimeMs": frame.get("videoTimeMs"),
        "frameNum": frame.get("frameNum"),
        "period": frame.get("period"),
        "game_event_id": frame.get("game_event_id"),
        "possession_event_id": frame.get("possession_event_id"),
    }

    game_event = frame.get("game_event", {})

    # Fall back to this frame's game_event only when the game-static ids were
    # not supplied, so the function stays usable on its own.
    if home_team_id is None and away_team_id is None and isinstance(game_event, dict):
        if game_event.get("home_team"):
            home_team_id = game_event.get("team_id")
        else:
            away_team_id = game_event.get("team_id")

    if isinstance(game_event, dict):
        for key in GAME_EVENT_KEYS:
            flat[key] = game_event.get(key, "")
    else:
        for key in GAME_EVENT_KEYS:
            flat[key] = ""

    possession_event = frame.get("possession_event", {})
    if isinstance(possession_event, dict):
        flat["possession_event_type"] = possession_event.get("possession_event_type", "")
    else:
        flat["possession_event_type"] = ""

    home_players = frame.get("homePlayers", [])
    away_players = frame.get("awayPlayers", [])

    if home_team_id:
        home_players = add_positions_to_players(home_players, home_team_id, position_map)
    if away_team_id:
        away_players = add_positions_to_players(away_players, away_team_id, position_map)

    flat["homePlayers"] = json.dumps(home_players if home_players else [])
    flat["homePlayersSmoothed"] = json.dumps(frame.get("homePlayersSmoothed", []))
    flat["awayPlayers"] = json.dumps(away_players if away_players else [])
    flat["awayPlayersSmoothed"] = json.dumps(frame.get("awayPlayersSmoothed", []))
    flat["balls"] = json.dumps(frame.get("balls", []))
    flat["ballsSmoothed"] = json.dumps(frame.get("ballsSmoothed", []))

    return flat


def extract_expanded_annotations(events_path):
    with open(events_path, "r") as f:
        data = json.load(f)

    annotations = []

    for event in data:
        labels = []

        possession = event.get("possessionEvents", {})
        event_type = possession.get("possessionEventType", "")
        body_type = possession.get("bodyType", "")

        if event_type == "PA":
            if body_type == "HE":
                labels.append("HEADER")
            elif possession.get("ballHeightType") == "A":
                labels.append("HIGH PASS")
            elif possession.get("passType") == "H":
                labels.append("THROW IN")
            else:
                labels.append("PASS")

        elif event_type == "CR":
            labels.append("CROSS")

        elif event_type == "SH":
            if body_type == "HE":
                labels.append("HEADER")
            labels.append("SHOT")
            if possession.get("shotOutcomeType") == "G":
                labels.append("GOAL")

        elif event_type == "CH":
            if possession.get("challengeWinnerPlayerId"):
                labels.append("PLAYER SUCCESSFUL TACKLE")

        elif event_type == "CL" and body_type == "HE":
            labels.append("HEADER")

        game_events = event.get("gameEvents", {})
        if game_events.get("gameEventType") == "OUT":
            labels.append("OUT")

        setpiece = game_events.get("setpieceType", "")
        if setpiece == "T":
            labels.append("THROW IN")
        elif setpiece == "F":
            labels.append("FREE KICK")

        for label in labels:
            if label in LABELS:
                annotation = {
                    "head": "action",
                    "label": label,
                    "position_ms": int(event.get("eventTime", 0) * 1000),
                    "gameTime": f"{game_events.get('period', 1)} - {game_events.get('startFormattedGameClock', '00:00')}",
                    "team": "home" if game_events.get("homeTeam", False) else "away",
                    "visibility": "visible",
                }
                annotations.append(annotation)

    return sorted(annotations, key=lambda x: x["position_ms"])


def deduplicate_annotations(annotations):
    """Keep one annotation per timestamp, ranked by LABEL_PRIORITY.

    The task is single-label, but the source stream emits several labels for a
    single moment: a throw-in is also a high ball, a free kick is also a pass,
    a headed shot is both HEADER and SHOT. LABEL_PRIORITY selects one.

    This is lossy across labels rather than uniformly, which matters when
    reading per-class results: HIGH PASS retains 89 of 2,697 candidates and
    SHOT 1,041 of 1,559, so those classes are sparse by construction rather
    than by data quality.
    """
    by_position = defaultdict(list)
    for ann in annotations:
        by_position[ann["position_ms"]].append(ann)

    deduped = []
    for position_ms in sorted(by_position.keys()):
        candidates = by_position[position_ms]
        if len(candidates) == 1:
            deduped.append(candidates[0])
        else:
            best = max(candidates, key=lambda a: LABEL_PRIORITY.get(a["label"], -1))
            deduped.append(best)

    return deduped


def filter_aligned_annotations(annotations, parquet_path, tolerance_ms):
    if not annotations:
        return annotations, 0

    tracking_df = pd.read_parquet(parquet_path, columns=["videoTimeMs", "frameNum"]).sort_values(
        ["videoTimeMs", "frameNum"], ascending=[True, True]
    ).reset_index(drop=True)

    kept = []
    skipped = 0

    for ann in annotations:
        position_ms = ann["position_ms"]
        time_diff = (tracking_df["videoTimeMs"] - position_ms).abs()
        closest_idx = time_diff.idxmin()

        if time_diff.loc[closest_idx] > tolerance_ms:
            skipped += 1
            continue

        kept.append(ann)

    return kept, skipped


def assign_splits(input_dir, suffix):
    files = sorted(f for f in os.listdir(input_dir) if f.endswith(suffix))
    split_map = {}

    for idx, filename in enumerate(files):
        game_id = filename[:-len(suffix)]
        for split, (start, end) in SPLITS.items():
            if start <= idx < end:
                split_map[game_id] = (split, idx)
                break

    return split_map
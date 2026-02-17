# soccernetpro/datasets/utils/tracking.py

"""
tracking utilities for player-position graph data.

provides constants, edge-building strategies, spatial augmentations,
and frame-level feature extraction for the SoccerNet-GAR tracking
modality.

feature vector layout (per object, per frame)::

    Index  Field
    -----  ----------------------------------
    0      x (pitch coordinates)
    1      y 
    2      is_ball (one-hot entity type)
    3      is_home
    4      is_away
    5      dx (velocity delta, computed across consecutive frames)
    6      dy 
    7      z (ball height; MISSING_VALUE for players)
"""

import json
import random

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# constants
# -------------------------------------------------------------------

NUM_OBJECTS = 23  # 1 ball + 11 home players + 11 away players
FEATURE_DIM = 8   # [x, y, is_ball, is_home, is_away, dx, dy, z]

# normalization bounds (pitch dimension in metres).
PITCH_HALF_LENGTH = 85.0
PITCH_HALF_WIDTH = 50.0
MAX_DISPLACEMENT = 110.0
MAX_BALL_HEIGHT = 30.0

# sentinel values used to mark missing / unobserved objects.
# raw features use MISSING_VALUE; after normalization they are mapped
# to MISSING_VALUE_NORMALIZED so the model sees a distinct, in-range
# indicator that cannot be confused with a valid coordinate.
MISSING_VALUE = -200.0
MISSING_VALUE_NORMALIZED = -2.0

# slot layout: index 0 is always the ball, 1-11 home, 12-22 away.
_BALL_SLOT = 0
_HOME_SLOT_START = 1
_HOME_SLOT_END = 12     # exclusive
_AWAY_SLOT_START = 12


# -------------------------------------------------------------------
# frame-level feature extraction
# -------------------------------------------------------------------

def parse_frame(row):
    """parse a single parquet row into per-object features and positions.

    Each row is expected to contain JSON-encoded columns balls,
    homePlayers, and awayPlayers following the SoccerNet-GAR
    tracking format.

    Args:
        row: A single row (pandas.Series) from a tracking parquet
            file.

    Returns:
        A tuple (features, positions) where features is a
        numpy.ndarray of shape (NUM_OBJECTS, FEATURE_DIM) and
        positions is a list of position-group strings (e.g.
        "GK", "DEF").  Unobserved slots are filled with
        MISSING_VALUE and an empty string respectively.
    """
    features = np.full(
        (NUM_OBJECTS, FEATURE_DIM), MISSING_VALUE, dtype=np.float32,
    )
    positions = [""] * NUM_OBJECTS

    obj_idx = _BALL_SLOT

    # -- ball (always slot 0) --
    ball_str = row.get("balls", "null")
    if pd.notna(ball_str) and ball_str not in ("null", ""):
        try:
            ball_list = json.loads(ball_str)
            if ball_list:
                ball = ball_list[0]
                x, y = ball.get("x"), ball.get("y")
                z = ball.get("z", 0)
                if x is not None and y is not None:
                    features[obj_idx] = [
                        float(x), float(y),
                        1, 0, 0,          # one-hot: ball
                        0, 0,             # dx, dy (filled later)
                        float(z),
                    ]
                    positions[obj_idx] = "BALL"
        except (json.JSONDecodeError, TypeError):
            pass
    obj_idx = _HOME_SLOT_START

    # -- home players (slots 1-11) --
    home_str = row.get("homePlayers", "[]")
    if pd.notna(home_str) and home_str not in ("null", ""):
        try:
            home_players = json.loads(home_str)
            home_players = sorted(
                home_players,
                key=lambda p: int(p.get("jerseyNum", 0)),
            )[:11]

            for player in home_players:
                x, y = player.get("x"), player.get("y")
                if x is not None and y is not None:
                    features[obj_idx] = [
                        float(x), float(y),
                        0, 1, 0,          # one-hot: home
                        0, 0,             # dx, dy
                        MISSING_VALUE,    # z unused for players
                    ]
                    positions[obj_idx] = player.get("positionGroup", "")
                obj_idx += 1

            # advance past any unfilled home slots.
            while obj_idx < _HOME_SLOT_END:
                obj_idx += 1
        except (json.JSONDecodeError, TypeError):
            obj_idx = _HOME_SLOT_END
    else:
        obj_idx = _HOME_SLOT_END

    # -- away players (slots 12-22) --
    away_str = row.get("awayPlayers", "[]")
    if pd.notna(away_str) and away_str not in ("null", ""):
        try:
            away_players = json.loads(away_str)
            away_players = sorted(
                away_players,
                key=lambda p: int(p.get("jerseyNum", 0)),
            )[:11]

            for player in away_players:
                x, y = player.get("x"), player.get("y")
                if x is not None and y is not None:
                    features[obj_idx] = [
                        float(x), float(y),
                        0, 0, 1,          # one-hot: away
                        0, 0,             # dx, dy
                        MISSING_VALUE,    # z unused for players
                    ]
                    positions[obj_idx] = player.get("positionGroup", "")
                obj_idx += 1
        except (json.JSONDecodeError, TypeError):
            pass

    return features, positions


# -------------------------------------------------------------------
# temporal feature computation
# -------------------------------------------------------------------

def compute_deltas(all_features):
    """compute per-object velocity deltas across consecutive frames.

    For each object that is observed in both frame t and frame t - 1,
    the displacement (dx, dy) is written into feature indices 5 and 6.
    frame 0 retains zero deltas.

    Args:
        all_features: numpy.ndarray of shape
            (num_frames, NUM_OBJECTS, FEATURE_DIM).

    Returns:
        the same array, modified in-place, with velocity deltas
        populated.
    """
    for t in range(1, all_features.shape[0]):
        for obj in range(NUM_OBJECTS):
            curr_valid = all_features[t, obj, 0] != MISSING_VALUE
            prev_valid = all_features[t - 1, obj, 0] != MISSING_VALUE
            if curr_valid and prev_valid:
                all_features[t, obj, 5] = (
                    all_features[t, obj, 0] - all_features[t - 1, obj, 0]
                )
                all_features[t, obj, 6] = (
                    all_features[t, obj, 1] - all_features[t - 1, obj, 1]
                )
    return all_features


def normalize_features(features):
    """normalize spatial features to roughly [-1, 1].

    observed coordinates are divided by the known pitch / displacement
    bounds.  unobserved slots are set to MISSING_VALUE_NORMALIZED
    so the model receives a distinct, out-of-range sentinel that
    cannot be confused with a valid normalized value.

    Args:
        features: numpy.ndarray of shape
            (num_frames, NUM_OBJECTS, FEATURE_DIM).

    Returns:
        a new array with the same shape containing normalized values.
        the input is not modified.
    """
    features_norm = features.copy()
    valid_mask = features_norm[:, :, 0] != MISSING_VALUE

    features_norm[valid_mask, 0] /= PITCH_HALF_LENGTH
    features_norm[valid_mask, 1] /= PITCH_HALF_WIDTH
    features_norm[valid_mask, 5] /= MAX_DISPLACEMENT
    features_norm[valid_mask, 6] /= MAX_DISPLACEMENT
    features_norm[valid_mask, 7] /= MAX_BALL_HEIGHT

    # write the normalized sentinel into every spatial channel of
    # missing objects so downstream layers can distinguish "absent"
    # from "at the origin".
    for ch in (0, 1, 5, 6, 7):
        features_norm[~valid_mask, ch] = MISSING_VALUE_NORMALIZED

    return features_norm


# -------------------------------------------------------------------
# edge building strategies
# -------------------------------------------------------------------

def build_edge_index(node_features, node_positions, edge_type, k=8, r=15.0):
    """build a graph edge index for a single frame.

    supports several connectivity strategies that trade off density
    against spatial or tactical priors.

    Args:
        node_features: numpy.ndarray of shape
            (NUM_OBJECTS, FEATURE_DIM).
        node_positions: List of position-group strings (length
            NUM_OBJECTS).
        edge_type: One of "none", "full", "knn",
            "distance", "ball_knn", "ball_distance",
            or "positional".
        k: Number of neighbours for knn / ball_knn strategies.
        r: Distance threshold (metres) for distance / ball_distance
            strategies.

    Returns:
        numpy.ndarray of shape (2, num_edges) in COO format,
        compatible with PyTorch Geometric.
    """
    if edge_type == "none":
        return np.zeros((2, 0), dtype=np.int64)

    if edge_type == "full":
        return _build_full_edges(node_features)

    if edge_type == "knn":
        return _build_knn_edges(node_features, k)

    if edge_type == "distance":
        return _build_distance_edges(node_features, threshold=r)

    if edge_type == "ball_knn":
        return _build_ball_knn_edges(node_features, k)

    if edge_type == "ball_distance":
        return _build_ball_distance_edges(node_features, threshold=r)

    if edge_type == "positional":
        return _build_positional_edges(node_features, node_positions)

    raise ValueError(f"Unknown edge_type: {edge_type}")


# -- strategy implementations (private) ----------------------------

def _build_full_edges(node_features):
    """fully connected graph, excluding self-loops and missing nodes."""
    num_nodes = node_features.shape[0]
    edge_list = [
        [i, j]
        for i in range(num_nodes)
        for j in range(num_nodes)
        if i != j
        and node_features[i, 0] != MISSING_VALUE
        and node_features[j, 0] != MISSING_VALUE
    ]

    if not edge_list:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(edge_list, dtype=np.int64).T


def _build_knn_edges(node_features, k):
    """k-nearest-neighbour edges based on Euclidean pitch distance."""
    num_nodes = node_features.shape[0]
    edge_list = []

    for i in range(num_nodes):
        if node_features[i, 0] == MISSING_VALUE:
            continue

        distances = []
        for j in range(num_nodes):
            if i != j and node_features[j, 0] != MISSING_VALUE:
                dist = np.linalg.norm(
                    node_features[i, :2] - node_features[j, :2],
                )
                distances.append((j, dist))

        distances.sort(key=lambda x: x[1])
        k_nearest = distances[: min(k, len(distances))]

        for neighbour_idx, _ in k_nearest:
            edge_list.append([i, neighbour_idx])
            edge_list.append([neighbour_idx, i])

    if not edge_list:
        return np.zeros((2, 0), dtype=np.int64)

    # de-duplicate symmetric pairs.
    edge_array = np.array(edge_list, dtype=np.int64).T
    edge_array = np.unique(edge_array, axis=1)
    return edge_array


def _build_distance_edges(node_features, threshold=15.0):
    """edges between all node pairs within a distance threshold."""
    num_nodes = node_features.shape[0]
    edge_list = []

    for i in range(num_nodes):
        if node_features[i, 0] == MISSING_VALUE:
            continue
        for j in range(i + 1, num_nodes):
            if node_features[j, 0] == MISSING_VALUE:
                continue
            dist = np.linalg.norm(
                node_features[i, :2] - node_features[j, :2],
            )
            if dist <= threshold:
                edge_list.append([i, j])
                edge_list.append([j, i])

    if not edge_list:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(edge_list, dtype=np.int64).T


def _build_positional_edges(node_features, node_positions):
    """tactical-structure edges following formation lines.

    connects adjacent lines within each team
    (GK <-> DEF <-> MID <-> FWD) and players within the same line.
    the ball is connected to every valid player on the pitch.
    """
    edge_list = []

    home_players = {}
    away_players = {}
    ball_idx = None

    for i in range(node_features.shape[0]):
        if node_features[i, 0] == MISSING_VALUE or not node_positions[i]:
            continue

        pos = node_positions[i]

        if pos == "BALL":
            ball_idx = i
        elif node_features[i, 3] == 1.0:  # home flag
            home_players.setdefault(pos, []).append(i)
        elif node_features[i, 4] == 1.0:  # away flag
            away_players.setdefault(pos, []).append(i)

    # intra-team tactical edges for both teams.
    for team_players in (home_players, away_players):
        gk = team_players.get("GK", [])
        defenders = team_players.get("DEF", [])
        midfielders = team_players.get("MID", [])
        forwards = team_players.get("FWD", [])

        # GK <-> DEF
        for p1 in gk:
            for p2 in defenders:
                edge_list.extend([[p1, p2], [p2, p1]])

        # DEF <-> DEF and DEF <-> MID
        for i, p1 in enumerate(defenders):
            for p2 in defenders[i + 1 :]:
                edge_list.extend([[p1, p2], [p2, p1]])
            for p2 in midfielders:
                edge_list.extend([[p1, p2], [p2, p1]])

        # MID <-> MID and MID <-> FWD
        for i, p1 in enumerate(midfielders):
            for p2 in midfielders[i + 1 :]:
                edge_list.extend([[p1, p2], [p2, p1]])
            for p2 in forwards:
                edge_list.extend([[p1, p2], [p2, p1]])

        # FWD <-> FWD
        for i, p1 in enumerate(forwards):
            for p2 in forwards[i + 1 :]:
                edge_list.extend([[p1, p2], [p2, p1]])

    # ball connects to every valid player on the pitch.
    if ball_idx is not None:
        for i in range(node_features.shape[0]):
            if i != ball_idx and node_features[i, 0] != MISSING_VALUE:
                edge_list.extend([[ball_idx, i], [i, ball_idx]])

    if not edge_list:
        return np.zeros((2, 0), dtype=np.int64)

    edge_array = np.array(edge_list, dtype=np.int64).T
    edge_array = np.unique(edge_array, axis=1)
    return edge_array


def _build_ball_knn_edges(node_features, k):
    """k nearest players to the ball, plus same-team interconnections.

    the ball node is connected to its k closest players.  Players
    among those k that share a team flag are also connected to each
    other (dot product of the one-hot team columns > 0).
    """
    ball_indices = np.where(node_features[:, 2] == 1.0)[0]
    if len(ball_indices) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    ball_idx = ball_indices[0]
    ball_pos = node_features[ball_idx, :2]

    if ball_pos[0] == MISSING_VALUE or ball_pos[1] == MISSING_VALUE:
        return np.zeros((2, 0), dtype=np.int64)

    num_nodes = node_features.shape[0]
    distances = []
    for i in range(num_nodes):
        is_player = (
            node_features[i, 3] == 1.0 or node_features[i, 4] == 1.0
        )
        if i != ball_idx and node_features[i, 0] != MISSING_VALUE and is_player:
            dist = np.linalg.norm(node_features[i, :2] - ball_pos)
            distances.append((i, dist))

    distances.sort(key=lambda x: x[1])
    k_nearest = distances[: min(k, len(distances))]

    edge_list = []

    # ball <-> each of the k nearest players.
    for player_idx, _ in k_nearest:
        edge_list.extend([[ball_idx, player_idx], [player_idx, ball_idx]])

    # same-team interconnections among the k nearest.
    k_nearest_indices = [idx for idx, _ in k_nearest]
    for i, idx_i in enumerate(k_nearest_indices):
        team_i = node_features[idx_i, 3:5]
        for j in range(i + 1, len(k_nearest_indices)):
            idx_j = k_nearest_indices[j]
            team_j = node_features[idx_j, 3:5]
            if np.dot(team_i, team_j) > 0:
                edge_list.extend([[idx_i, idx_j], [idx_j, idx_i]])

    if not edge_list:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(edge_list, dtype=np.int64).T


def _build_ball_distance_edges(node_features, threshold=20.0):
    """players within a distance threshold of the ball, plus same-team edges.

    every player within threshold metres of the ball receives a
    bidirectional edge to the ball node.  players among those that
    share a team flag are also interconnected.
    """
    ball_indices = np.where(node_features[:, 2] == 1.0)[0]
    if len(ball_indices) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    ball_idx = ball_indices[0]
    ball_pos = node_features[ball_idx, :2]

    if ball_pos[0] == MISSING_VALUE or ball_pos[1] == MISSING_VALUE:
        return np.zeros((2, 0), dtype=np.int64)

    num_nodes = node_features.shape[0]
    edge_list = []
    nearby_players = []

    for i in range(num_nodes):
        is_player = (
            node_features[i, 3] == 1.0 or node_features[i, 4] == 1.0
        )
        if i != ball_idx and node_features[i, 0] != MISSING_VALUE and is_player:
            dist = np.linalg.norm(node_features[i, :2] - ball_pos)
            if dist <= threshold:
                edge_list.extend([[ball_idx, i], [i, ball_idx]])
                nearby_players.append(i)

    # same-team interconnections among nearby players.
    for i, idx_i in enumerate(nearby_players):
        team_i = node_features[idx_i, 3:5]
        for j in range(i + 1, len(nearby_players)):
            idx_j = nearby_players[j]
            team_j = node_features[idx_j, 3:5]
            if np.dot(team_i, team_j) > 0:
                edge_list.extend([[idx_i, idx_j], [idx_j, idx_i]])

    if not edge_list:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(edge_list, dtype=np.int64).T


# -------------------------------------------------------------------
# augmentations
# -------------------------------------------------------------------

class HorizontalFlip:
    """randomly negate x-coordinates and dx-velocities (pitch length axis).

    Args:
        probability: chance of applying the flip per call.
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, features):
        """apply the transform.

        Args:
            features: numpy.ndarray of shape
                (num_frames, NUM_OBJECTS, FEATURE_DIM).

        Returns:
            the (possibly flipped) feature array. a copy is made when
            the flip is applied; the input is never modified.
        """
        if random.random() < self.probability:
            features_flipped = features.copy()
            valid_mask = features_flipped[:, :, 0] != MISSING_VALUE
            features_flipped[valid_mask, 0] *= -1  # flip x
            features_flipped[valid_mask, 5] *= -1  # flip dx
            return features_flipped
        return features


class VerticalFlip:
    """randomly negate y-coordinates and dy-velocities (pitch width axis).

    Args:
        probability: chance of applying the flip per call.
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, features):
        """apply the transform.

        Args:
            features: numpy.ndarray of shape
                (num_frames, NUM_OBJECTS, FEATURE_DIM).

        Returns:
            the (possibly flipped) feature array.
        """
        if random.random() < self.probability:
            features_flipped = features.copy()
            valid_mask = features_flipped[:, :, 0] != MISSING_VALUE
            features_flipped[valid_mask, 1] *= -1  # flip y
            features_flipped[valid_mask, 6] *= -1  # flip dy
            return features_flipped
        return features


class TeamFlip:
    """randomly swap the home and away one-hot team flags.

    this is a label-preserving augmentation: swapping team identity
    does not change the group activity class.

    Args:
        probability: chance of applying the swap per call.
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, features):
        """apply the transform.

        Args:
            features: numpy.ndarray of shape
                (num_frames, NUM_OBJECTS, FEATURE_DIM).

        Returns:
            the (possibly swapped) feature array.
        """
        if random.random() < self.probability:
            features_flipped = features.copy()
            home_col = features_flipped[:, :, 3].copy()
            features_flipped[:, :, 3] = features_flipped[:, :, 4]
            features_flipped[:, :, 4] = home_col
            return features_flipped
        return features

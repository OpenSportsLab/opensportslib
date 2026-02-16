# soccernetpro/datasets/utils/tracking.py
"""
Tracking utilities for player position data.
Includes edge building strategies and augmentation transforms.
"""

import random
import numpy as np


# ============================================================
# Constants
# ============================================================

NUM_OBJECTS = 23  # 1 ball + 11 home + 11 away
FEATURE_DIM = 8   # x, y, is_ball, home, away, dx, dy, z

# Normalization bounds
PITCH_HALF_LENGTH = 85.0
PITCH_HALF_WIDTH = 50.0
MAX_DISPLACEMENT = 110.0
MAX_BALL_HEIGHT = 30.0


# ============================================================
# Edge Building
# ============================================================

def build_edge_index(node_features, node_positions, edge_type, k=8):
    """
    Build edge index for a single frame based on edge_type.
    
    Args:
        node_features: np.array of shape (NUM_OBJECTS, FEATURE_DIM)
        node_positions: list of position group strings
        edge_type: str, one of ['none', 'full', 'knn', 'distance', 'positional']
        k: int, number of neighbors for knn
        
    Returns:
        edge_index: np.array of shape (2, num_edges)
    """
    if edge_type == 'none':
        return np.zeros((2, 0), dtype=np.int64)
    
    if edge_type == 'full':
        return _build_full_edges(node_features)
    
    if edge_type == 'knn':
        return _build_knn_edges(node_features, k)
    
    if edge_type == 'distance':
        return _build_distance_edges(node_features)
    
    if edge_type == 'positional':
        return _build_positional_edges(node_features, node_positions)
    
    raise ValueError(f"Unknown edge_type: {edge_type}")


def _build_full_edges(node_features):
    """Fully connected graph (excluding self-loops and invalid nodes)."""
    num_nodes = node_features.shape[0]
    edge_list = [
        [i, j] 
        for i in range(num_nodes) 
        for j in range(num_nodes) 
        if i != j and node_features[i, 0] != -200.0 and node_features[j, 0] != -200.0
    ]
    
    if not edge_list:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(edge_list, dtype=np.int64).T


def _build_knn_edges(node_features, k):
    """K-nearest neighbors edges."""
    num_nodes = node_features.shape[0]
    edge_list = []
    
    for i in range(num_nodes):
        if node_features[i, 0] == -200.0:
            continue
        
        distances = []
        for j in range(num_nodes):
            if i != j and node_features[j, 0] != -200.0:
                dist = np.linalg.norm(node_features[i, :2] - node_features[j, :2])
                distances.append((j, dist))
        
        distances.sort(key=lambda x: x[1])
        k_nearest = distances[:min(k, len(distances))]
        
        for neighbor_idx, _ in k_nearest:
            edge_list.append([i, neighbor_idx])
            edge_list.append([neighbor_idx, i])
    
    if not edge_list:
        return np.zeros((2, 0), dtype=np.int64)
    
    edge_array = np.array(edge_list, dtype=np.int64).T
    edge_array = np.unique(edge_array, axis=1)
    return edge_array


def _build_distance_edges(node_features, threshold=15.0):
    """Distance threshold edges."""
    num_nodes = node_features.shape[0]
    edge_list = []
    
    for i in range(num_nodes):
        if node_features[i, 0] == -200.0:
            continue
        for j in range(i + 1, num_nodes):
            if node_features[j, 0] == -200.0:
                continue
            dist = np.linalg.norm(node_features[i, :2] - node_features[j, :2])
            if dist <= threshold:
                edge_list.append([i, j])
                edge_list.append([j, i])
    
    if not edge_list:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(edge_list, dtype=np.int64).T


def _build_positional_edges(node_features, node_positions):
    """
    Tactical structure edges: GK <-> DEF <-> MID <-> FWD.
    Ball connects to all players.
    """
    edge_list = []
    
    # Group players by team and position
    home_players = {}
    away_players = {}
    ball_idx = None
    
    for i in range(node_features.shape[0]):
        if node_features[i, 0] == -200.0 or not node_positions[i]:
            continue
        
        pos = node_positions[i]
        
        if pos == 'BALL':
            ball_idx = i
        elif node_features[i, 3] == 1.0:  # home team
            if pos not in home_players:
                home_players[pos] = []
            home_players[pos].append(i)
        elif node_features[i, 4] == 1.0:  # away team
            if pos not in away_players:
                away_players[pos] = []
            away_players[pos].append(i)
    
    # Build tactical edges for each team
    for team_players in [home_players, away_players]:
        gk = team_players.get('GK', [])
        defenders = team_players.get('DEF', [])
        midfielders = team_players.get('MID', [])
        forwards = team_players.get('FWD', [])
        
        # GK <-> DEF
        for p1 in gk:
            for p2 in defenders:
                edge_list.extend([[p1, p2], [p2, p1]])
        
        # DEF <-> DEF and DEF <-> MID
        for i, p1 in enumerate(defenders):
            for p2 in defenders[i+1:]:
                edge_list.extend([[p1, p2], [p2, p1]])
            for p2 in midfielders:
                edge_list.extend([[p1, p2], [p2, p1]])
        
        # MID <-> MID and MID <-> FWD
        for i, p1 in enumerate(midfielders):
            for p2 in midfielders[i+1:]:
                edge_list.extend([[p1, p2], [p2, p1]])
            for p2 in forwards:
                edge_list.extend([[p1, p2], [p2, p1]])
        
        # FWD <-> FWD
        for i, p1 in enumerate(forwards):
            for p2 in forwards[i+1:]:
                edge_list.extend([[p1, p2], [p2, p1]])
    
    # Ball connects to all players
    if ball_idx is not None:
        for i in range(node_features.shape[0]):
            if i != ball_idx and node_features[i, 0] != -200.0:
                edge_list.extend([[ball_idx, i], [i, ball_idx]])
    
    if not edge_list:
        return np.zeros((2, 0), dtype=np.int64)
    
    edge_array = np.array(edge_list, dtype=np.int64).T
    edge_array = np.unique(edge_array, axis=1)
    return edge_array


# ============================================================
# Augmentations
# ============================================================

class HorizontalFlip:
    """Randomly flips x coordinates horizontally (along pitch length)."""
    
    def __init__(self, probability=0.5):
        self.probability = probability
    
    def __call__(self, features):
        if random.random() < self.probability:
            features_flipped = features.copy()
            valid_mask = features_flipped[:, :, 0] != -200.0
            features_flipped[valid_mask, 0] *= -1  # flip x
            features_flipped[valid_mask, 5] *= -1  # flip dx
            return features_flipped
        return features


class VerticalFlip:
    """Randomly flips y coordinates vertically (along pitch width)."""
    
    def __init__(self, probability=0.5):
        self.probability = probability
    
    def __call__(self, features):
        if random.random() < self.probability:
            features_flipped = features.copy()
            valid_mask = features_flipped[:, :, 0] != -200.0
            features_flipped[valid_mask, 1] *= -1  # flip y
            features_flipped[valid_mask, 6] *= -1  # flip dy
            return features_flipped
        return features


class TeamFlip:
    """Randomly swaps home and away team labels."""
    
    def __init__(self, probability=0.5):
        self.probability = probability
    
    def __call__(self, features):
        if random.random() < self.probability:
            features_flipped = features.copy()
            home_team = features_flipped[:, :, 3].copy()
            features_flipped[:, :, 3] = features_flipped[:, :, 4]
            features_flipped[:, :, 4] = home_team
            return features_flipped
        return features
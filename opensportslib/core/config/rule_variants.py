"""Shared rule-variant metadata, independent of model dependencies."""

HEADER_RULE_VARIANTS = {
    "h5_header_distance": {
        "trajectory_filter_enabled": False,
        "trajectory_change_mode": "either_angle_or_speed",
        "created_by": "h5_header_distance_rule",
    },
    "h5_header_distance_speed": {
        "trajectory_filter_enabled": True,
        "trajectory_change_mode": "speed",
        "created_by": "h5_header_distance_speed_rule",
    },
    "h5_header_distance_angle": {
        "trajectory_filter_enabled": True,
        "trajectory_change_mode": "angle",
        "created_by": "h5_header_distance_angle_rule",
    },
    "h5_header_distance_speed_angle": {
        "trajectory_filter_enabled": True,
        "trajectory_change_mode": "both_angle_and_speed",
        "created_by": "h5_header_distance_speed_angle_rule",
    },
}


SKELETON_RULE_VARIANTS = {
    "h5_header_skeleton": {},
    # Recall-first: keep only the gates that cost (almost) no true headers.
    # The trajectory gates and the narrow height band together reject about a
    # tenth of real headers -- flick-ons and glancing contacts barely disturb
    # the ball -- so they are dropped. The hand check is kept because it is
    # nearly free in recall and removes many arm and keeper contacts, and the
    # wider suppression window merges the extra detections each duel produces.
    "h5_header_skeleton_recall": {
        # Any tracked head joint counts, not just the nose: a nose is often
        # untracked while an ear or the neck is, and skipping those players
        # loses contacts outright.
        "head_joints": ["nose", "neck", "l_ear", "r_ear", "l_eye", "r_eye"],
        "velocity_change_min_mps": 0.0,
        "velocity_mag_min_mps": 0.0,
        # A 10 degree bend is the one trajectory test worth keeping here. It
        # costs 2.9 points of recall and returns 8 of precision: 97.1/69.4 with
        # it, 100.0/61.4 without. Set to 0.0 when a missed header matters more
        # than a false one; that finds every annotated header on the 2022 final.
        "angle_change_min_deg": 10.0,
        "accel_z_change_min_mps2": 0.0,
        "incoming_speed_min_mps": 0.0,
        "ball_height_min_m": 0.5,
        "ball_height_max_m": 8.0,
        "ankle_height_max_m": 99.0,
        "dwell_max_frames": 999999,
        "nms_window_frames": 40,
        "created_by": "h5_header_skeleton_recall_rule",
    },
}

# Recall taken as far as it goes: the same settings with the bend test off.
# On the 2022 final that finds every annotated header, 100% recall at 61.4%
# precision, against 97.1/69.4 with the bend test. Two in five predictions are
# then wrong, so this is for building a candidate set something else filters.
SKELETON_RULE_VARIANTS["h5_header_skeleton_max_recall"] = {
    **SKELETON_RULE_VARIANTS["h5_header_skeleton_recall"],
    "angle_change_min_deg": 0.0,
    "created_by": "h5_header_skeleton_max_recall_rule",
}

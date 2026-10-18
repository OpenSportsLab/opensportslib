"""Shared preprocessing configuration without importing inference runtimes."""
from opensportslib.core.config.accessors import get_input_cfg, get_data_params, get_data_sampling


def is_skeleton_config(config):
    return get_input_cfg(config).get('representation') == 'skeleton_graph'


def preprocessing_config(config):
    cfg = dict(get_data_params(config).get('preprocessing', {}))
    cfg['sample_fps'] = float(get_data_sampling(config).get('extract_fps', 25))
    return cfg


def validate_preprocessing(cfg):
    if cfg.get('backend', 'rtmlib') != 'rtmlib':
        raise ValueError('Only the rtmlib pose backend is supported')
    if cfg.get('type', 'video_to_skeleton') != 'video_to_skeleton':
        raise ValueError('skeleton_graph requires video_to_skeleton preprocessing')
    for key, default in [('max_players', 2), ('max_candidates', 8)]:
        if not isinstance(cfg.get(key, default), int) or cfg.get(key, default) < 1:
            raise ValueError(f'{key} must be a positive integer')
    if cfg.get('max_candidates', 8) < cfg.get('max_players', 2):
        raise ValueError('max_candidates must be at least max_players')
    if cfg.get('sample_fps', 25) <= 0:
        raise ValueError('sample_fps must be positive')
    if cfg.get('track_retention_seconds', .5) < 0:
        raise ValueError('track_retention_seconds must be nonnegative')
    if cfg.get('selection', 'tennis_conservative') not in {'tennis_conservative', 'all_people'}:
        raise ValueError('Unknown player selection policy')
    mode = cfg.get('cache', {}).get('mode', 'off')
    if mode not in {'off', 'read'}:
        raise ValueError('Inference cache mode must be off or read; use extract() to write')

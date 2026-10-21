"""Shared preprocessing configuration without importing inference runtimes."""
from opensportslib.core.config.accessors import get_component_by_kind, get_data_sampling


def pose_component(config):
    if isinstance(config, dict):
        components = config.get('MODEL', {}).get('components', {})
        component = next((c for c in components.values() if c.get('kind') == 'preprocessor'), None)
    else:
        component = get_component_by_kind(config, 'preprocessor')
    if component and component.get('source', {}).get('name') == 'pose_tracking':
        return component
    return None


def requires_raw_video(config):
    """Input capability lookup; does not import torch or a pose runtime."""
    return pose_component(config) is not None


def preprocessing_config(config):
    component = pose_component(config)
    if component is None:
        raise ValueError('Define MODEL.components.pose_tracking with kind=preprocessor')
    cfg = dict(component.get('params', {}))
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
    if mode is False:  # YAML 1.1 treats an unquoted `off` as boolean false.
        cfg['cache']['mode'] = mode = 'off'
    if mode not in {'off', 'read', 'read_write'}:
        raise ValueError('Cache mode must be off, read or read_write; use extract to warm caches')
    if mode != 'off' and not cfg.get('cache', {}).get('path'):
        raise ValueError('Enabled clip caching requires cache.path')

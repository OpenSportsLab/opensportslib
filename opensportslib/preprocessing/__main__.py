"""Explicit extraction, overlay, and full-pipeline benchmark commands."""
import argparse


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    warm=sub.add_parser('warm-cache', help='Extract exact clips through the existing video loader')
    warm.add_argument('--config',required=True)
    warm.add_argument('--output',required=True)
    warm.add_argument('--split',default='train',choices=['train','valid','test'])
    warm.add_argument('--max-clips',type=int)
    for name in ('extract','benchmark','overlay'):
        command=sub.add_parser(name)
        command.add_argument('--video',required=True)
        command.add_argument('--output',required=True)
        if name=='overlay':command.add_argument('--cache',required=True)
        else:command.add_argument('--config',required=True)
        if name=='benchmark':command.add_argument('--repeats',type=int,default=3)
        if name=='benchmark':command.add_argument('--view',action='append',default=[],help='Additional classification camera view')
    args=parser.parse_args(argv)
    if args.command=='overlay':
        from .visualization import render_overlay
        render_overlay(args.video,args.cache,args.output)
    else:
        from opensportslib.core.config.editable import Config
        from opensportslib.core.config.runtime_adapter import dict_to_namespace
        config=dict_to_namespace(Config.from_file(args.config).get_config())
        from opensportslib.core.utils.config import resolve_config_omega
        config=resolve_config_omega(config)
        if args.command=='warm-cache':
            from .cache import warm_clip_cache
            warm_clip_cache(config,args.output,split=args.split,max_clips=args.max_clips)
        elif args.command=='extract':
            from .session import extract
            from .config import preprocessing_config
            extract(args.video,args.output,preprocessing_config(config))
        else:
            from .benchmark import benchmark
            benchmark(config,args.video,args.output,args.repeats,view_paths=args.view)
    return 0


if __name__=='__main__':
    raise SystemExit(main())

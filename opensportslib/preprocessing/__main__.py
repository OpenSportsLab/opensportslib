"""Explicit extraction, overlay, and full-pipeline benchmark commands."""
import argparse


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('extract','benchmark','overlay'):
        command=sub.add_parser(name)
        command.add_argument('--video',required=True)
        command.add_argument('--output',required=True)
        if name=='overlay':command.add_argument('--cache',required=True)
        else:command.add_argument('--config',required=True)
        if name=='benchmark':command.add_argument('--repeats',type=int,default=3)
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
        if args.command=='extract':
            from .session import extract
            from .config import preprocessing_config
            extract(args.video,args.output,preprocessing_config(config))
        else:
            from .benchmark import benchmark
            benchmark(config,args.video,args.output,args.repeats)
    return 0


if __name__=='__main__':
    raise SystemExit(main())

"""
Script for creating dataset based on exisiting labels.
"""
from argparse import ArgumentParser
from .datasets.creator import create_dataset, _DEFAULT_VAR_TO_FILE_INFO
from .utils.config import save_var_to_file_info

def create_action(args):
    """
    Run the create action.
    """
    if args.var is None:
        args.var = ['mms1_dis_dist_fast']

    trange = [args.start, args.end]
    kwargs = {
        'force': args.force,
        'samples': args.samples,
        'clean': args.clean,
        'var_list': args.var,
        'resample': args.resample,
        'label_source': args.label_source,
        'var_info_file': args.var_info_file,
    }

    create_dataset(args.output, trange, **kwargs)


def pars_args():
    """
    Parse commandline arguments.
    """
    parser = ArgumentParser()

    actions = parser.add_subparsers(dest="command")

    export = actions.add_parser('export', help='Export config files.')
    export.add_argument('--var-info-file', default="var_to_file_info.toml")

    create = actions.add_parser('create', help='Create a dataset')
    create.add_argument('--label_source', default='Olshevsky',
                        choices=['Olshevsky', 'Unlabeled'])
    create.add_argument('--start', default='2017-11-01',
                        help='Start date, format YYYY-MM-DD/HH:MM:DD')
    create.add_argument('--end', default='2017-11-30',
                        help='End date, format YYYY-MM-DD/HH:MM:DD')
    create.add_argument('--force', action='store_true', default=False)
    create.add_argument('--clean', action='store_true', default=False)
    create.add_argument('--samples', default=0)
    create.add_argument('--resample', default=None)
    create.add_argument('--var',
                        action='append',
                        choices=_DEFAULT_VAR_TO_FILE_INFO.keys())
    create.add_argument('--var-info-file', default=None)
    create.add_argument('output')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        exit(1)

    print("Arguments:")
    for arg in vars(args):
        print(f" {arg}: {getattr(args, arg)}")

    return args

def main():
    """
    Main function for the SpacePhyML CLI.
    """
    args = pars_args()
    if args.command == 'create':
        create_action(args)
    elif args.command == 'export':
        save_var_to_file_info(_DEFAULT_VAR_TO_FILE_INFO, args.var_info_file)

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import argparse
import logging

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

log = logging.getLogger(name=__name__)


def graph_rewards(data, output_path):
    plt.stackplot(
        range(len(data)),
        -500 * data.missed.values,
        data.sloth.values,
        colors=['r', 'b']
    )
    plt.xlim((0, len(data) - 1))
    plt.title('Cost per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Cost')

    sloth = mpatches.Patch(color='b', label='Due to Elapsed Time')
    miss = mpatches.Patch(color='r', label='Due to Missed Slaloms')
    plt.legend(loc=4, handles=[miss, sloth])

    plt.savefig(output_path, format='png')


def parse_args():
    """
    Parses the arguments from the command line

    Returns
    -------
    argparse.Namespace
    """
    desc = 'Plot the per-episode rewards'
    parser = argparse.ArgumentParser(description=desc)

    data_path_help = 'Path to csv with "reward", "sloth", and "missed" columns'
    parser.add_argument('data_path',
                        type=str,
                        help=data_path_help)

    output_path_help = 'Where to save the plot as pdf'
    parser.add_argument('output_path',
                        type=str,
                        help=output_path_help)

    # Parse the command line arguments
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data = pd.read_csv(args.data_path, index_col=0)
    graph_rewards(data, args.output_path)

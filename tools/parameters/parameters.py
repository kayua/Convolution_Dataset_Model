#!/usr/bin/python3
# -*- coding: utf-8 -*-

DEFAULT_SNAPSHOT_COLUMN_POSITION = 1
DEFAULT_PEER_COLUMN_POSITION = 2
DEFAULT_FEATURE_WINDOW_LENGTH = 256
DEFAULT_FEATURE_WINDOW_WIDTH = 256
DEFAULT_NUMBER_BLOCK_PER_SAMPLES = 32
DEFAULT_INPUT_FILE_SWARM_SORTED = 'S4'
DEFAULT_OUTPUT_FILE_SWARM_SORTED = 'S4_output.txt'


def add_arguments(parser):


    help_msg = 'Snapshot column position (Default {})'.format(DEFAULT_SNAPSHOT_COLUMN_POSITION)
    parser.add_argument("--column_snapshot", type=int, help=help_msg, default=DEFAULT_SNAPSHOT_COLUMN_POSITION)

    help_msg = 'Peer column position (Default {})'.format(DEFAULT_PEER_COLUMN_POSITION)
    parser.add_argument("--column_peer", type=int, help=help_msg, default=DEFAULT_PEER_COLUMN_POSITION)

    help_msg = 'Feature window length (Default {})'.format(DEFAULT_FEATURE_WINDOW_LENGTH)
    parser.add_argument("--window_length", type=int, help=help_msg, default=DEFAULT_FEATURE_WINDOW_LENGTH)

    help_msg = 'Feature window width (Default {})'.format(DEFAULT_FEATURE_WINDOW_WIDTH)
    parser.add_argument("--window_width", type=int, help=help_msg, default=DEFAULT_FEATURE_WINDOW_WIDTH)

    help_msg = 'Number block per samples (Default {})'.format(DEFAULT_NUMBER_BLOCK_PER_SAMPLES)
    parser.add_argument("--number_block", type=int, help=help_msg, default=DEFAULT_NUMBER_BLOCK_PER_SAMPLES)

    help_msg = 'Input swarm file (Default {})'.format(DEFAULT_INPUT_FILE_SWARM_SORTED)
    parser.add_argument("--input_swarm", type=str, help=help_msg, default=DEFAULT_INPUT_FILE_SWARM_SORTED)

    help_msg = 'Output swarm file (Default {})'.format(DEFAULT_OUTPUT_FILE_SWARM_SORTED)
    parser.add_argument("--output_swarm", type=str, help=help_msg, default=DEFAULT_OUTPUT_FILE_SWARM_SORTED)

    return parser







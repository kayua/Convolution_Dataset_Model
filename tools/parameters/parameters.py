#!/usr/bin/python3
# -*- coding: utf-8 -*-

DEFAULT_LENGTH_WINDOW_FEATURE = 256
DEFAULT_WIDTH_WINDOW_FEATURE = 256

def add_arguments(parser):


    help_msg = 'Feature window length (Default {})'.format(DEFAULT_FILE_INPUT_SWARM_UNSORTED)
    parser.add_argument("--input_unsorted_file", type=str, help=help_msg, default=DEFAULT_FILE_INPUT_SWARM_UNSORTED)


    return parser







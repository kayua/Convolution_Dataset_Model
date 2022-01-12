#!/usr/bin/python3
# -*- coding: utf-8 -*-

DEFAULT_SNAPSHOT_COLUMN_POSITION = 1
DEFAULT_PEER_COLUMN_POSITION = 2
DEFAULT_FEATURE_WINDOW_LENGTH = 256
DEFAULT_FEATURE_WINDOW_WIDTH = 256

self.break_point = 1
self.matrix_features = []
self.number_block_per_samples = 32
self.input_file_swarm_sorted = 'S4'
self.output_file_swarm_sorted = 'S4_output.txt'
self.features = []
self.input_feature = []
self.feature_input = []
self.snapshot_id = self.feature_window_length

def add_arguments(parser):


    help_msg = 'Feature window length (Default {})'.format(DEFAULT_FILE_INPUT_SWARM_UNSORTED)
    parser.add_argument("--input_unsorted_file", type=str, help=help_msg, default=DEFAULT_FILE_INPUT_SWARM_UNSORTED)


    return parser







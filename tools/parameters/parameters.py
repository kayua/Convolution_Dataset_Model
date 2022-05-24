#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
from sys import argv

DEFAULT_SNAPSHOT_COLUMN_POSITION = 1
DEFAULT_PEER_COLUMN_POSITION = 2
DEFAULT_FEATURE_WINDOW_LENGTH = 256
DEFAULT_FEATURE_WINDOW_WIDTH = 256
DEFAULT_NUMBER_BLOCK_PER_SAMPLES = 32
DEFAULT_NEURAL_TOPOLOGY = 'model_v1'
DEFAULT_ADVERSARIAL_MODEL = False
DEFAULT_VERBOSITY = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'

DEFAULT_INPUT_FILE_SWARM = ''
DEFAULT_SAVE_FILE_SWARM = ''
DEFAULT_LOAD_SAMPLES_TRAINING_IN = ''
DEFAULT_LOAD_SAMPLES_OUT = ''

DEFAULT_TRAINING_EPOCHS = 120
DEFAULT_TRAINING_METRICS = 'mse'
DEFAULT_TRAINING_LOSS = 'mse'
DEFAULT_TRAINING_OPTIMIZER = 'adam'
DEFAULT_TRAINING_BATCH_SIZE = 32
DEFAULT_SAVE_MODEL_FILE = 'models_saved/model'
DEFAULT_TRAINING_THRESHOLD = 0.75
DEFAULT_TRAINING_SEED = 0
DEFAULT_TRAINING_PIF = 0
DEFAULT_LEARNING_RATE = 0.001

DEFAULT_PREDICT_LOAD_MODEL_FILE = ''
DEFAULT_INPUT_SWARM_FILE = ''
DEFAULT_OUTPUT_SWARM_FILE = ''
DEFAULT_CORRECTED_SWARM_FILE = ''
DEFAULT_FAILED_SWARM_FILE = ''
DEFAULT_ORIGINAL_SWARM_FILE = ''
DEFAULT_FILE_ANALYSE_MODE = '+a'
DEFAULT_FILE_ANALYSE_RESULTS = 'results.txt'


def show_config(args):

    logging.info('Command:\n\t{0}\n'.format(' '.join([x for x in argv])))
    logging.info('Settings:')
    lengths = [len(x) for x in vars(args).keys()]
    max_lengths = max(lengths)

    for parameters, item_args in sorted(vars(args).items()):
        message = "\t"
        message += parameters.ljust(max_lengths, ' ')
        message += ' : {}'.format(item_args)
        logging.info(message)

    logging.info("")


def dataset_arguments(parser):

    help_msg = 'Input file swarm (Default {})'.format(DEFAULT_INPUT_FILE_SWARM)
    parser.add_argument("--input_file_swarm", type=str, help=help_msg, default=DEFAULT_INPUT_FILE_SWARM)

    help_msg = 'Save file samples (Default {})'.format(DEFAULT_SAVE_FILE_SWARM)
    parser.add_argument("--save_file_samples", type=str, help=help_msg, default=DEFAULT_SAVE_FILE_SWARM)

    help_msg = 'Load file samples in (Default {})'.format(DEFAULT_LOAD_SAMPLES_TRAINING_IN)
    parser.add_argument("--load_samples_in", type=str, help=help_msg, default=DEFAULT_LOAD_SAMPLES_TRAINING_IN)

    help_msg = 'Load file samples out (Default {})'.format(DEFAULT_LOAD_SAMPLES_OUT)
    parser.add_argument("--load_samples_out", type=str, help=help_msg, default=DEFAULT_LOAD_SAMPLES_OUT)

    help_msg = 'File save model (Default {})'.format(DEFAULT_SAVE_MODEL_FILE)
    parser.add_argument("--save_model", type=str, help=help_msg, default=DEFAULT_SAVE_MODEL_FILE)

    help_msg = 'File load model (Default {})'.format(DEFAULT_PREDICT_LOAD_MODEL_FILE)
    parser.add_argument("--load_model", type=str, help=help_msg, default=DEFAULT_PREDICT_LOAD_MODEL_FILE)

    help_msg = 'File input to predict (Default {})'.format(DEFAULT_INPUT_SWARM_FILE)
    parser.add_argument("--input_predict", type=str, help=help_msg, default=DEFAULT_INPUT_SWARM_FILE)

    help_msg = 'File output to predict (Default {})'.format(DEFAULT_OUTPUT_SWARM_FILE)
    parser.add_argument("--output_predict", type=str, help=help_msg, default=DEFAULT_OUTPUT_SWARM_FILE)

    help_msg = 'File corrected for evaluation (Default {})'.format(DEFAULT_CORRECTED_SWARM_FILE)
    parser.add_argument("--file_corrected", type=str, help=help_msg, default=DEFAULT_CORRECTED_SWARM_FILE)

    help_msg = 'File failed for evaluation (Default {})'.format(DEFAULT_FAILED_SWARM_FILE)
    parser.add_argument("--file_failed", type=str, help=help_msg, default=DEFAULT_FAILED_SWARM_FILE)

    help_msg = 'File failed for evaluation (Default {})'.format(DEFAULT_ORIGINAL_SWARM_FILE)
    parser.add_argument("--file_original", type=str, help=help_msg, default=DEFAULT_ORIGINAL_SWARM_FILE)

    help_msg = 'File evaluation file mode (Default {})'.format(DEFAULT_FILE_ANALYSE_MODE)
    parser.add_argument("--file_analyse_mode", type=str, help=help_msg, default=DEFAULT_FILE_ANALYSE_MODE)

    help_msg = 'File evaluation file (Default {})'.format(DEFAULT_FILE_ANALYSE_RESULTS)
    parser.add_argument("--file_analyse", type=str, help=help_msg, default=DEFAULT_FILE_ANALYSE_RESULTS)

    return parser


def dataset_parameters(parser):

    help_msg = 'Snapshot column position (Default {})'.format(DEFAULT_SNAPSHOT_COLUMN_POSITION)
    parser.add_argument("--snapshot_column", type=int, help=help_msg, default=DEFAULT_SNAPSHOT_COLUMN_POSITION)

    help_msg = 'Peer column position (Default {})'.format(DEFAULT_PEER_COLUMN_POSITION)
    parser.add_argument("--peer_column", type=int, help=help_msg, default=DEFAULT_PEER_COLUMN_POSITION)

    help_msg = 'Define length window (Default {})'.format(DEFAULT_FEATURE_WINDOW_LENGTH)
    parser.add_argument("--window_length", type=int, help=help_msg, default=DEFAULT_FEATURE_WINDOW_LENGTH)

    help_msg = 'Define width window (Default {})'.format(DEFAULT_FEATURE_WINDOW_WIDTH)
    parser.add_argument("--window_width", type=int, help=help_msg, default=DEFAULT_FEATURE_WINDOW_WIDTH)

    help_msg = 'Define number blocks (Default {})'.format(DEFAULT_NUMBER_BLOCK_PER_SAMPLES)
    parser.add_argument("--topo_version", type=int, help=help_msg, default=DEFAULT_NUMBER_BLOCK_PER_SAMPLES)

    help_msg = 'Neural topology (Default {})'.format(DEFAULT_NEURAL_TOPOLOGY)
    parser.add_argument("--topology", type=str, help=help_msg, default=DEFAULT_NEURAL_TOPOLOGY)

    help_msg = 'Verbosity (Default {})'.format(DEFAULT_VERBOSITY)
    parser.add_argument("--verbosity", type=int, help=help_msg, default=DEFAULT_VERBOSITY)

    help_msg = 'Define number epochs (Default {})'.format(DEFAULT_TRAINING_EPOCHS)
    parser.add_argument("--epochs", type=int, help=help_msg, default=DEFAULT_TRAINING_EPOCHS)

    help_msg = 'Define metrics (Default {})'.format(DEFAULT_TRAINING_METRICS)
    parser.add_argument("--metrics", type=str, help=help_msg, default=DEFAULT_TRAINING_METRICS)

    help_msg = 'Define loss (Default {})'.format(DEFAULT_TRAINING_LOSS)
    parser.add_argument("--loss", type=str, help=help_msg, default=DEFAULT_TRAINING_LOSS)

    help_msg = 'Define optimizer (Default {})'.format(DEFAULT_TRAINING_OPTIMIZER)
    parser.add_argument("--optimizer", type=str, help=help_msg, default=DEFAULT_TRAINING_OPTIMIZER)

    help_msg = 'Define batch size (Default {})'.format(DEFAULT_TRAINING_BATCH_SIZE)
    parser.add_argument("--steps_per_epoch", type=int, help=help_msg, default=DEFAULT_TRAINING_BATCH_SIZE)

    help_msg = 'Threshold (Default {})'.format(DEFAULT_TRAINING_THRESHOLD)
    parser.add_argument("--threshold", type=float, help=help_msg, default=DEFAULT_TRAINING_THRESHOLD)

    help_msg = 'Seed (Default {})'.format(DEFAULT_TRAINING_SEED)
    parser.add_argument("--seed", type=int, help=help_msg, default=DEFAULT_TRAINING_SEED)

    help_msg = 'Learning rate (Default {})'.format(DEFAULT_LEARNING_RATE)
    parser.add_argument("--learning_rate", type=int, help=help_msg, default=DEFAULT_LEARNING_RATE)

    help_msg = 'PIF (Default {})'.format(DEFAULT_TRAINING_PIF)
    parser.add_argument("--pfi", type=int, help=help_msg, default=DEFAULT_TRAINING_PIF)

    return parser


def add_arguments(parser):

    parser = dataset_parameters(parser)
    parser = dataset_arguments(parser)
    cmd_choices = ['Calibration', 'CreateSamples', 'Training', 'Predict', 'Analyse']
    parser.add_argument('cmd', choices=cmd_choices)
    return parser

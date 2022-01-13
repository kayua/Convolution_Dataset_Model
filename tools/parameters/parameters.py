#!/usr/bin/python3
# -*- coding: utf-8 -*-

DEFAULT_SNAPSHOT_COLUMN_POSITION = 1
DEFAULT_PEER_COLUMN_POSITION = 2
DEFAULT_FEATURE_WINDOW_LENGTH = 256
DEFAULT_FEATURE_WINDOW_WIDTH = 256
DEFAULT_NUMBER_BLOCK_PER_SAMPLES = 32
DEFAULT_NEURAL_TOPOLOGY = 'ModelV1'
DEFAULT_ADVERSARIAL_MODEL = False
DEFAULT_VERBOSITY = 1
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'

DEFAULT_CREATE_INPUT_FILE_SWARM_IN = ''
DEFAULT_CREATE_INPUT_FILE_SWARM_OUT = ''

DEFAULT_TRAINING_EPOCHS = 10
DEFAULT_TRAINING_METRICS = 'mse'
DEFAULT_TRAINING_LOSS = 'mse'
DEFAULT_TRAINING_OPTIMIZER = 'adam'
DEFAULT_TRAINING_BATCH_SIZE = 32
DEFAULT_TRAINING_SAVE_MODEL_FILE = 'models_saved/model'
DEFAULT_TRAINING_THRESHOLD = 1.0
DEFAULT_TRAINING_SEED = 0
DEFAULT_TRAINING_PIF = 0

DEFAULT_PREDICT_LOAD_MODEL_FILE = 'models_saved/model'
DEFAULT_PREDICT_INPUT_SWARM_FILE = ''
DEFAULT_PREDICT_OUTPUT_SWARM_FILE = ''

DEFAULT_EVALUATION_CORRECTED_SWARM_FILE = ''
DEFAULT_EVALUATION_FAILED_SWARM_FILE = ''
DEFAULT_EVALUATION_ORIGINAL_SWARM_FILE = ''
DEFAULT_EVALUATION_FILE_ANALYSE_MODE = '+a'
DEFAULT_EVALUATION_FILE_ANALYSE_RESULTS = 'results.txt'


def add_arguments(parser):


    help_msg = 'Snapshot column position (Default {})'.format(DEFAULT_SNAPSHOT_COLUMN_POSITION)
    parser.add_argument("--snapshot_column", type=int, help=help_msg, default=DEFAULT_SNAPSHOT_COLUMN_POSITION)

    help_msg = 'Peer column position (Default {})'.format(DEFAULT_PEER_COLUMN_POSITION)
    parser.add_argument("--peer_column", type=int, help=help_msg, default=DEFAULT_PEER_COLUMN_POSITION)

    help_msg = 'Define length window (Default {})'.format(DEFAULT_FEATURE_WINDOW_LENGTH)
    parser.add_argument("--window_length", type=int, help=help_msg, default=DEFAULT_FEATURE_WINDOW_LENGTH)

    help_msg = 'Define width window (Default {})'.format(DEFAULT_FEATURE_WINDOW_WIDTH)
    parser.add_argument("--window_width", type=int, help=help_msg, default=DEFAULT_FEATURE_WINDOW_WIDTH)

    help_msg = 'Define number blocks (Default {})'.format(DEFAULT_NUMBER_BLOCK_PER_SAMPLES)
    parser.add_argument("--number_blocks", type=int, help=help_msg, default=DEFAULT_NUMBER_BLOCK_PER_SAMPLES)

    help_msg = 'Neural topology (Default {})'.format(DEFAULT_NEURAL_TOPOLOGY)
    parser.add_argument("--topology", type=str, help=help_msg, default=DEFAULT_NEURAL_TOPOLOGY)

    help_msg = 'Adversarial mode (Default {})'.format(DEFAULT_ADVERSARIAL_MODEL)
    parser.add_argument("--adversarial", type=bool, help=help_msg, default=DEFAULT_ADVERSARIAL_MODEL)

    help_msg = 'Verbosity (Default {})'.format(DEFAULT_VERBOSITY)
    parser.add_argument("--verbosity", type=int, help=help_msg, default=DEFAULT_VERBOSITY)

    help_msg = 'Input file create samples (INPUT NEURAL) (Default {})'.format(DEFAULT_CREATE_INPUT_FILE_SWARM_IN)
    parser.add_argument("--input_file_swarm", type=str, help=help_msg, default=DEFAULT_CREATE_INPUT_FILE_SWARM_IN)

    help_msg = 'Input file create samples (OUTPUT NEURAL) (Default {})'.format(DEFAULT_CREATE_INPUT_FILE_SWARM_OUT)
    parser.add_argument("--file_save_samples", type=str, help=help_msg, default=DEFAULT_CREATE_INPUT_FILE_SWARM_OUT)

    help_msg = 'Define number epochs (Default {})'.format(DEFAULT_TRAINING_EPOCHS)
    parser.add_argument("--epochs", type=str, help=help_msg, default=DEFAULT_TRAINING_EPOCHS)

    help_msg = 'Define metrics (Default {})'.format(DEFAULT_TRAINING_METRICS)
    parser.add_argument("--metrics", type=str, help=help_msg, default=DEFAULT_TRAINING_METRICS)

    help_msg = 'Define loss (Default {})'.format(DEFAULT_TRAINING_LOSS)
    parser.add_argument("--loss", type=str, help=help_msg, default=DEFAULT_TRAINING_LOSS)

    help_msg = 'Define optimizer (Default {})'.format(DEFAULT_TRAINING_OPTIMIZER)
    parser.add_argument("--optimizer", type=str, help=help_msg, default=DEFAULT_TRAINING_OPTIMIZER)

    help_msg = 'Define batch size (Default {})'.format(DEFAULT_TRAINING_BATCH_SIZE)
    parser.add_argument("--steps_per_epochs", type=str, help=help_msg, default=DEFAULT_TRAINING_BATCH_SIZE)

    help_msg = 'File save model (Default {})'.format(DEFAULT_TRAINING_SAVE_MODEL_FILE)
    parser.add_argument("--save_model", type=str, help=help_msg, default=DEFAULT_TRAINING_SAVE_MODEL_FILE)

    help_msg = 'File load model (Default {})'.format(DEFAULT_PREDICT_LOAD_MODEL_FILE)
    parser.add_argument("--load_model", type=str, help=help_msg, default=DEFAULT_PREDICT_LOAD_MODEL_FILE)

    help_msg = 'File input to predict (Default {})'.format(DEFAULT_PREDICT_INPUT_SWARM_FILE)
    parser.add_argument("--input_predict", type=str, help=help_msg, default=DEFAULT_PREDICT_INPUT_SWARM_FILE)

    help_msg = 'File output to predict (Default {})'.format(DEFAULT_PREDICT_OUTPUT_SWARM_FILE)
    parser.add_argument("--output_predict", type=str, help=help_msg, default=DEFAULT_PREDICT_OUTPUT_SWARM_FILE)

    help_msg = 'File corrected for evaluation (Default {})'.format(DEFAULT_EVALUATION_CORRECTED_SWARM_FILE)
    parser.add_argument("--file_corrected", type=str, help=help_msg, default=DEFAULT_EVALUATION_CORRECTED_SWARM_FILE)

    help_msg = 'File failed for evaluation (Default {})'.format(DEFAULT_EVALUATION_FAILED_SWARM_FILE)
    parser.add_argument("--file_failed", type=str, help=help_msg, default=DEFAULT_EVALUATION_FAILED_SWARM_FILE)

    help_msg = 'File failed for evaluation (Default {})'.format(DEFAULT_EVALUATION_ORIGINAL_SWARM_FILE)
    parser.add_argument("--file_original", type=str, help=help_msg, default=DEFAULT_EVALUATION_ORIGINAL_SWARM_FILE)

    help_msg = 'File analyse file mode (Default {})'.format(DEFAULT_EVALUATION_FILE_ANALYSE_MODE)
    parser.add_argument("--file_analyse_mode", type=str, help=help_msg, default=DEFAULT_EVALUATION_FILE_ANALYSE_MODE)

    help_msg = 'File analyse file (Default {})'.format(DEFAULT_EVALUATION_FILE_ANALYSE_RESULTS)
    parser.add_argument("--file_analyse", type=str, help=help_msg, default=DEFAULT_EVALUATION_FILE_ANALYSE_RESULTS)

    help_msg = 'Threshold (Default {})'.format(DEFAULT_TRAINING_THRESHOLD)
    parser.add_argument("--threshold", type=float, help=help_msg, default=DEFAULT_TRAINING_THRESHOLD)

    help_msg = 'Seed (Default {})'.format(DEFAULT_TRAINING_SEED)
    parser.add_argument("--seed", type=int, help=help_msg, default=DEFAULT_TRAINING_SEED)

    help_msg = 'PIF (Default {})'.format(DEFAULT_TRAINING_PIF)
    parser.add_argument("--pfi", type=int, help=help_msg, default=DEFAULT_TRAINING_PIF)

    cmd_choices = ['Calibration', 'CreateSamples', 'Training', 'Predict', 'Analyse']
    parser.add_argument('cmd', choices=cmd_choices)
    return parser


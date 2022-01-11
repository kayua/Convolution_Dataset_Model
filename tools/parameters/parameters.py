#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = ' @gmail.com, @unipampa.edu.br '
__version__ = '{2}.{0}.{1}'
__data__ = '2021/11/21'
__credits__ = ['All']

DEFAULT_FILE_INPUT_SWARM_UNSORTED = ''
DEFAULT_FILE_OUTPUT_SWARM_SORTED = ''

DEFAULT_FILE_INPUT_SWARM_SORTED = 'dataset/processed/S1a'
DEFAULT_FILE_SAVE_SWARM_SAMPLES = ''
DEFAULT_FILE_LOAD_SWARM_SAMPLES_TRAINING_X = ''
DEFAULT_FILE_LOAD_SWARM_SAMPLES_TRAINING_Y = ''
DEFAULT_FILE_LOAD_SWARM_SAMPLES_PREDICT = ''
DEFAULT_FEATURE_WINDOW_WIDTH = 128
DEFAULT_FEATURE_WINDOW_LENGTH = 128
DEFAULT_SNAPSHOT_COLUMN_IDENTIFIER = 1
DEFAULT_PEER_COLUMN_IDENTIFIER = 2
DEFAULT_TRAINING_EPOCHS = 5
DEFAULT_TRAINING_METRICS = 'binary_crossentropy'
DEFAULT_TRAINING_LOSS = 'binary_crossentropy'
DEFAULT_TRAINING_OPTIMIZER = 'adam'
DEFAULT_STEPS_PER_EPOCH = 32
DEFAULT_SAVE_MODEL = 'models_saved/model'
DEFAULT_LOAD_MODEL = 'models_saved/model'
DEFAULT_VERBOSITY = 10
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'


def add_arguments(parser):

    # Ordenar arquivos
    help_msg = 'Define input unsorted file swarm_corrected (Default {})'.format(DEFAULT_FILE_INPUT_SWARM_UNSORTED)
    parser.add_argument("--input_unsorted_file", type=str, help=help_msg, default=DEFAULT_FILE_INPUT_SWARM_UNSORTED)

    help_msg = 'Define output sorted file swarm_corrected (Default {})'.format(DEFAULT_FILE_OUTPUT_SWARM_SORTED)
    parser.add_argument("--output_sorted_file", type=str, help=help_msg, default=DEFAULT_FILE_OUTPUT_SWARM_SORTED)

    # Criar arquivo de amostras
    help_msg = 'File input swarm_corrected sorted (Default {})'.format(DEFAULT_FILE_INPUT_SWARM_SORTED)
    parser.add_argument("--input_file_swarm_sorted", type=str, help=help_msg, default=DEFAULT_FILE_INPUT_SWARM_SORTED)

    help_msg = 'File input swarm_corrected sorted (Default {})'.format(DEFAULT_FILE_INPUT_SWARM_SORTED)
    parser.add_argument("--output_file_samples", type=str, help=help_msg, default=DEFAULT_FILE_INPUT_SWARM_SORTED)


    # Treinar modelo de rede neural
    help_msg = 'File input swarm_corrected sorted (Default {})'.format(DEFAULT_FILE_INPUT_SWARM_SORTED)
    parser.add_argument("--input_samples_training_in", type=str, help=help_msg, default=DEFAULT_FILE_LOAD_SWARM_SAMPLES_TRAINING_X)

    help_msg = 'File input swarm_corrected sorted (Default {})'.format(DEFAULT_FILE_INPUT_SWARM_SORTED)
    parser.add_argument("--input_samples_training_out", type=str, help=help_msg, default=DEFAULT_FILE_LOAD_SWARM_SAMPLES_TRAINING_Y)

    help_msg = 'File input swarm_corrected sorted (Default {})'.format(DEFAULT_FILE_INPUT_SWARM_SORTED)
    parser.add_argument("--input_samples_predict", type=str, help=help_msg, default=DEFAULT_FILE_LOAD_SWARM_SAMPLES_PREDICT)

    help_msg = 'Correcting width window (Default {})'.format(DEFAULT_FEATURE_WINDOW_WIDTH)
    parser.add_argument("--epochs", type=int, help=help_msg, default=DEFAULT_TRAINING_EPOCHS)

    help_msg = 'Correcting width window (Default {})'.format(DEFAULT_FEATURE_WINDOW_WIDTH)
    parser.add_argument("--metrics", type=str, help=help_msg, default='mse')

    help_msg = 'Correcting width window (Default {})'.format(DEFAULT_FEATURE_WINDOW_WIDTH)
    parser.add_argument("--loss", type=str, help=help_msg, default='binary_crossentropy')

    help_msg = 'Correcting width window (Default {})'.format(DEFAULT_FEATURE_WINDOW_WIDTH)
    parser.add_argument("--optimizer", type=str, help=help_msg, default='adam')

    help_msg = 'Correcting width window (Default {})'.format(DEFAULT_FEATURE_WINDOW_WIDTH)
    parser.add_argument("--steps_per_epoch", type=int, help=help_msg, default=32)

    help_msg = 'Correcting width window (Default {})'.format(DEFAULT_FEATURE_WINDOW_WIDTH)
    parser.add_argument("--saved_models", type=str, help=help_msg, default='models')

    help_msg = 'Correcting width window (Default {})'.format(DEFAULT_FEATURE_WINDOW_WIDTH)
    parser.add_argument("--adversarial_model", type=bool, help=help_msg, default=True)

    help_msg = 'Correcting width window (Default {})'.format(DEFAULT_FEATURE_WINDOW_WIDTH)
    parser.add_argument("--load_model", type=str, help=help_msg, default='models')

    help_msg = 'Correcting width window (Default {})'.format(DEFAULT_FEATURE_WINDOW_WIDTH)
    parser.add_argument("--file_load_model", type=str, help=help_msg, default='')

    help_msg = 'Correcting width window (Default {})'.format(DEFAULT_FEATURE_WINDOW_WIDTH)
    parser.add_argument("--neural_model", type=str, help=help_msg, default='model_v1a')

    # Parametros de ajuste
    help_msg = 'Correcting width window (Default {})'.format(DEFAULT_FEATURE_WINDOW_WIDTH)
    parser.add_argument("--width_window", type=str, help=help_msg, default=DEFAULT_FEATURE_WINDOW_WIDTH)

    help_msg = 'Correcting length window (Default {})'.format(DEFAULT_FEATURE_WINDOW_LENGTH)
    parser.add_argument("--length_window", type=str, help=help_msg, default=DEFAULT_FEATURE_WINDOW_LENGTH)

    help_msg = 'Snapshot column position (Default {})'.format(DEFAULT_SNAPSHOT_COLUMN_IDENTIFIER)
    parser.add_argument("--snapshot_position", type=int, help=help_msg, default=DEFAULT_SNAPSHOT_COLUMN_IDENTIFIER)

    help_msg = 'Peer column position (Default {})'.format(DEFAULT_PEER_COLUMN_IDENTIFIER)
    parser.add_argument("--peer_position", type=int, help=help_msg, default=DEFAULT_PEER_COLUMN_IDENTIFIER)
    
    help_msg = 'Peer column position (Default {})'.format(DEFAULT_PEER_COLUMN_IDENTIFIER)
    parser.add_argument("--file_results", type=str, help=help_msg, default='')

    help_msg = 'Set verbosity (Default {})'.format(DEFAULT_VERBOSITY)
    parser.add_argument("--verbosity", type=int, help=help_msg, default=DEFAULT_VERBOSITY)

    help_msg = 'Set verbosity (Default {})'.format(DEFAULT_VERBOSITY)
    parser.add_argument("--analyse_file", type=str, help=help_msg, default=DEFAULT_VERBOSITY)

    help_msg = 'Set verbosity (Default {})'.format(DEFAULT_VERBOSITY)
    parser.add_argument("--original_file_swarm", type=str, help=help_msg, default=DEFAULT_VERBOSITY)

    help_msg = 'Set verbosity (Default {})'.format(DEFAULT_VERBOSITY)
    parser.add_argument("--corrected_file_swarm", type=str, help=help_msg, default=DEFAULT_VERBOSITY)

    help_msg = 'Set verbosity (Default {})'.format(DEFAULT_VERBOSITY)
    parser.add_argument("--failed_file_swarm", type=str, help=help_msg, default=DEFAULT_VERBOSITY)

    cmd_choices = ['SortDataset', 'CreateSamples', 'Training', 'Calibration', 'Predict', 'Analyse']
    parser.add_argument('cmd', choices=cmd_choices)

    return parser







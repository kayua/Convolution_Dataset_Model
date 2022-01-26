import logging
from argparse import ArgumentParser
from sys import argv

from models.neural import Neural
from models.neural_models.model_v1 import ModelsV1
from tools.analyse.analyse import Analyse
from tools.dataset.dataset import Dataset
from tools.parameters.parameters import add_arguments
from tools.parameters.parameters import TIME_FORMAT


def calibration_neural_model(args):

    logging.info('Starting calibration')
    neural_network_instance = create_classifier_model(args)
    neural_network_instance.calibration_neural_network()
    logging.info('End calibration')


def create_samples(args):

    logging.info('Creating file samples')
    dataset_instance = Dataset(args)
    dataset_instance.load_swarm_to_feature()
    dataset_instance.save_file_samples_features()
    logging.info('Samples file created')


def training_neural_model(args):

    logging.info('Start training neural network model')
    dataset_instance_input = Dataset(args)
    dataset_instance_output = Dataset(args)
    neural_network_instance = create_classifier_model(args)
    dataset_instance_input.load_file_samples(args.load_samples_training_in)
    dataset_instance_output.load_file_samples(args.load_samples_training_out)
    training_input_samples = dataset_instance_input.get_features()
    training_output_samples = dataset_instance_output.get_features()
    neural_network_instance.training(training_input_samples, training_output_samples)
    neural_network_instance.save_network()
    logging.info('End training neural network model')


def predict_neural_model(args):

    logging.info('Start prediction neural network model')
    dataset_instance_input = Dataset(args)
    neural_network_instance = create_classifier_model(args)
    dataset_instance_input.load_file_samples(args.input_predict)
    predict_input_samples = dataset_instance_input.get_features()
    features_predicted = neural_network_instance.predict(predict_input_samples)
    dataset_instance_input.cast_all_features_to_swarm(features_predicted, predict_input_samples)
    logging.info('End prediction neural network model')


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


def create_classifier_model(args):

    neural_model = Neural(args)

    if args.cmd == 'Predict':
        neural_model.load_model()
        return neural_model

    if args.cmd == 'Training' or args.cmd == 'Calibration':
        neural_model.create_neural_network(ModelsV1(args))
        return neural_model


def evaluation(args):

    evaluation_model = Analyse(args)
    evaluation_model.get_all_metrics()
    evaluation_model.write_results_analyse()


def arguments_cmd_choice(args):

    if args.cmd == 'Calibration': calibration_neural_model(args)
    if args.cmd == 'CreateSamples': create_samples(args)
    if args.cmd == 'Training': training_neural_model(args)
    if args.cmd == 'Predict': predict_neural_model(args)
    if args.cmd == 'Analyse': evaluation(args)


def main():

    parser = ArgumentParser(description='Correct trace adversarial model')
    parser = add_arguments(parser)
    args = parser.parse_args()

    if args.verbosity == logging.DEBUG:
        logging.basicConfig(format="%(asctime)s %(levelname)s {%(module)s} [%(funcName)s] %(message)s",
                            datefmt=TIME_FORMAT, level=args.verbosity)
        show_config(args)

    else:
        logging.basicConfig(format="%(message)s", datefmt=TIME_FORMAT, level=args.verbosity)

    arguments_cmd_choice(args)


if __name__ == '__main__':
    exit(main())

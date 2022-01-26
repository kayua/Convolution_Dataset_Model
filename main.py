import logging
from argparse import ArgumentParser

from models.neural import Neural
from models.neural_models.model_v1 import ModelsV1
from tools.analyse.analyse import Analyse
from tools.dataset.dataset import Dataset
from tools.parameters.parameters import add_arguments, show_config
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


def create_classifier_model(args):

    neural_model = Neural(args)

    if args.cmd == 'Predict':
        neural_model.load_model()
        return neural_model

    if args.cmd == 'Training' or args.cmd == 'Calibration':

        if args.topology == 'model_v1':
            neural_model.create_neural_network(ModelsV1(args))

        if args.topology == 'model_v2':
            neural_model.create_neural_network(ModelsV1(args))

        return neural_model


def evaluation(args):

    logging.info('Start evaluation neural network model')
    evaluation_model = Analyse(args)
    evaluation_model.get_all_metrics()
    evaluation_model.write_results_analyse()
    logging.info('End evaluation neural network model')


def arguments_cmd_choice(args):

    if args.cmd == 'Calibration': calibration_neural_model(args)
    if args.cmd == 'CreateSamples': create_samples(args)
    if args.cmd == 'Training': training_neural_model(args)
    if args.cmd == 'Predict': predict_neural_model(args)
    if args.cmd == 'Analyse': evaluation(args)


def main():

    argument_parser = ArgumentParser(description='Regenerating Datasets With Convolutional Network')
    argument_parser = add_arguments(argument_parser)
    arguments = argument_parser.parse_args()

    if arguments.verbosity == logging.DEBUG:
        logging.basicConfig(format="%(asctime)s %(levelname)s {%(module)s} [%(funcName)s] %(message)s",
                            datefmt=TIME_FORMAT, level=arguments.verbosity)
        show_config(arguments)

    else:

        logging.basicConfig(format="%(message)s", datefmt=TIME_FORMAT, level=arguments.verbosity)

    arguments_cmd_choice(arguments)


if __name__ == '__main__':
    exit(main())

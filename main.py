#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = ' @gmail.com, @unipampa.edu.br '
__version__ = '{2}.{0}.{2}'
__data__ = '2021/11/21'
__credits__ = ['All']


try:
    import sys
    import logging
    import subprocess
    import shlex
    from argparse import ArgumentParser
    from models.neural import Neural
    from models.neural_models.model_v1 import ModelsV1
    from models.neural_models.model_v2 import ModelsV2
    from models.neural_models.model_v3 import ModelsV3
    from models.neural_models.model_v3 import ModelsV4
    from tools.evaluation.analyse import Analyse
    from tools.dataset.dataset import Dataset
    from tools.parameters.parameters import add_arguments, show_config
    from tools.parameters.parameters import TIME_FORMAT
    import numpy

except ImportError as error:

    print(error)
    print()
    print("1. Setup a virtual environment: ")
    print("  python3 - m venv ~/Python3env/Mosquitoes ")
    print("  source ~/Python3env/Regenerating_dataset/bin/activate ")
    print()
    print("2. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    exit(-1)


def calibration_neural_model(args):

    logging.info('Starting calibration')
    neural_network_instance = create_classifier_model(args)
    neural_network_instance.calibration_neural_network()
    logging.info('End calibration')


def create_samples(args):

    logging.info('Creating file samples')
    dataset_instance = Dataset(args)
    logging.debug("create_samples: a")
    dataset_instance.load_swarm_to_feature()
    logging.debug("create_samples: b")
    dataset_instance.save_file_samples_features()
    logging.info('Samples file created')


def training_neural_model(args):

    logging.info('Start training neural network model')

    dataset_instance_input = Dataset(args)
    logging.debug('dataset_instance_input : {}'.format(dataset_instance_input))

    dataset_instance_output = Dataset(args)
    logging.debug('dataset_instance_output: {}'.format(dataset_instance_output))

    neural_network_instance = create_classifier_model(args)
    dataset_instance_input.load_file_samples(args.load_samples_in)
    logging.debug('args.load_samples_in  : {}'.format(args.load_samples_in))

    dataset_instance_output.load_file_samples(args.load_samples_out)
    logging.debug('args.load_samples_out : {}'.format(args.load_samples_out))

    training_input_samples = dataset_instance_input.get_features()
    training_output_samples = dataset_instance_output.get_features()
    neural_network_instance.training(training_input_samples, training_output_samples)
    neural_network_instance.save_network()
    logging.info('End training neural network model')

    
def merge_samples(args):
    merge_samples_in = None
    merge_samples_out = None

    logging.info('Merge samples')
    logging.info("\tOUTPUT")
    dataset_out = Dataset(args)
    dataset_out.load_file_samples(args.load_samples_out)

    logging.info("\tINPUT")
    files_in = args.load_samples_in.split(";")
    logging.debug("\tfiles_in: {}".format(files_in))
    for f in files_in:
        logging.info("\t\tfile: {}".format(f))
        dataset_in = Dataset(args)
        dataset_in.load_file_samples(f)
        if merge_samples_in is None:
            merge_samples_in = dataset_in.get_features()
            merge_samples_out = dataset_out.get_features()
        else:
            merge_samples_in += dataset_in.get_features()
            merge_samples_out += dataset_out.get_features()

    logging.info("\n\tWriting file")
    try:
        logging.info("\t\tinput: {}".format(args.save_file_samples))
        numpy.savez(args.save_file_samples, merge_samples_in)

        logging.info("\t\toutut: {}_output".format(args.save_file_samples))
        numpy.savez("{}_output".format(args.save_file_samples), merge_samples_out)

    except FileNotFoundError:
        logging.error('Error: writing file error: {}'.format(args.save_file_samples))
        sys.exit(-1)
    logging.info("\tDone!")
    # dataset_instance_input_1 = Dataset(args) # 1% falha
    # logging.debug('dataset_instance_input : {}'.format(dataset_instance_input))
    #
    # dataset_instance_output_1 = Dataset(args) # 1% falha
    # logging.debug('dataset_instance_output: {}'.format(dataset_instance_output))
    #
    #
    # dataset_instance_input_2 = Dataset(args)   # 10% falha
    # logging.debug('dataset_instance_input : {}'.format(dataset_instance_input))
    #
    # dataset_instance_output_2 = Dataset(args) # 10% falha
    # logging.debug('dataset_instance_output: {}'.format(dataset_instance_output))
    #
    # dataset_instance_input_2.load_file_samples(args.load_samples_in)
    # logging.debug('args.load_samples_in  : {}'.format(args.load_samples_in))
    #
    # dataset_instance_input_2.load_file_samples(args.load_samples_out)
    # logging.debug('args.load_samples_out : {}'.format(args.load_samples_out))
    #
    # training_input_samples = dataset_instance_input_1.get_features()
    # training_output_samples = dataset_instance_output_1.get_features()
    #
    # training_input_samples += dataset_instance_input_2.get_features()
    # training_output_samples += dataset_instance_output_2.get_features()
    
    #dataset_instance.save_file_samples_features() #Verificar como esse método gera o npz
    

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

    #TODO: melhorar essa API
    if args.load_model is not None:
        neural_model.load_model()

    else:
        if args.cmd == 'Training' or args.cmd == 'Calibration':

            if args.topology == 'model_v1':
                neural_model.create_neural_network(ModelsV1(args))

            if args.topology == 'model_v2':
                neural_model.create_neural_network(ModelsV2(args))

            if args.topology == 'model_v3':
                neural_model.create_neural_network(ModelsV3(args))

            if args.topology == 'model_v4':
                neural_model.create_neural_network(ModelsV4(args))

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
    #batch de treinamento
    if args.cmd == 'Training': training_neural_model(args)
    if args.cmd == 'Predict': predict_neural_model(args)
    if args.cmd == 'Analyse': evaluation(args)

    if args.cmd == 'MergeSamples': merge_samples(args)


def imprime_config(args):
    '''
    Mostra os argumentos recebidos e as configurações processadas
    :args: parser.parse_args
    '''
    logging.info("Command:\n\t{0}\n".format(" ".join([x for x in sys.argv])))
    logging.info("Settings:")
    lengths = [len(x) for x in vars(args).keys()]
    max_length = max(lengths)

    for k, v in sorted(vars(args).items()):
        message = "\t"
        message += k.ljust(max_length, " ")
        message += " : {}".format(v)
        logging.info(message)

    logging.info("")


def run_cmd(cmd_str, shell=False, check=True):
    logging.debug("Cmd_str: {}".format(cmd_str))
    # transforma em array por questões de segurança -> https://docs.python.org/3/library/shlex.html
    cmd_array = shlex.split(cmd_str)
    logging.debug("Cmd_array: {}".format(cmd_array))
    # executa comando em subprocesso
    subprocess.run(cmd_array, check=check, shell=shell)


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

    imprime_config(arguments)

    cmd = "mkdir -p samples_saved/samples_training_in/"
    run_cmd(cmd)

    cmd = "mkdir -p samples_saved/samples_training_out/"
    run_cmd(cmd)

    cmd = "mkdir -p samples_saved/samples_predict/"
    run_cmd(cmd)

    cmd = "mkdir -p models_saved/model"
    run_cmd(cmd)

    arguments_cmd_choice(arguments)


if __name__ == '__main__':
    exit(main())

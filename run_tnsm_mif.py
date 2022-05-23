#!/usr/bin/python3
# -*- coding: utf-8 -*-

try:
    import sys
    import os
    from tqdm import tqdm
    import argparse
    import logging
    import subprocess
    import shlex
    import datetime
    from logging.handlers import RotatingFileHandler

except ImportError as error:
    print(error)
    print()
    print("1. (optional) Setup a virtual environment: ")
    print("  python3 - m venv ~/Python3env/mltraces ")
    print("  source ~/Python3env/mltraces/bin/activate ")
    print()
    print("2. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

#https://liyin2015.medium.com/tensorflow-cpus-and-gpus-configuration-9c223436d4ef
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

DEFAULT_OUTPUT_FILE = "default_mfi_ouput.txt"
DEFAULT_APPEND_OUTPUT_FILE = False
DEFAULT_VERBOSITY_LEVEL = logging.INFO
DEFAULT_TRIALS = 1
DEFAULT_START_TRIALS = 0
DEFAULT_CAMPAIGN = "demo"
DEFAULT_VALIDATION_DATASET = "swarm/validation/S1_25.sort_u_1n_3n"
DEFAULT_TRAINING_DATASET = "S2a"
NUM_EPOCHS = 120
TIME_FORMAT = '%Y-%m-%d_%H:%M:%S'

PATH_ORIGINAL = "data/01_original"
PATH_TRAINING = "data/01_original"
PATH_FAILED_MON = "data/02_failed_monitors"
PATH_FAILED_PROB = "data/02_failed_probability"

PATH_RANKING = "data/02_failed_monitors/ranking"
PATH_CORRECTED = "data/03_corrected_monitors"
PATH_MODEL = "models_saved"
PATH_LOG = 'logs/'
PATHS = [PATH_ORIGINAL, PATH_TRAINING, PATH_FAILED_MON, PATH_FAILED_PROB, PATH_CORRECTED, PATH_MODEL, PATH_LOG]

SOURCE_ZIP_FILES = "/home/mansilha/Research/acdc/"
files = ["00_Collection_TRACE_RES-100_from-w5000-to-w6000.zip",
         "00_Collection_TRACE_RES-100.zip",
         "01_Aerofly_TRACE_RES-100.zip",
         "02_Increibles_TRACE_RES-100.zip",
         "03_Happytime_TRACE_RES-100.zip",
         "04_Star_TRACE_RES-100.zip",
         "05_Mission_TRACE_RES-100.zip",
         "02_Increibles_TRACE_RES-100_tail_99pct.zip"]

training_files = []
args = None


def get_output_file_name(campaign=DEFAULT_CAMPAIGN):
    return "results_noms22_mfi_{}.txt".format(campaign)


def print_config(args):

    logging.info("Command:\n\t{0}\n".format(" ".join([x for x in sys.argv])))
    logging.info("Settings:")
    lengths = [len(x) for x in vars(args).keys()]
    max_lenght = max(lengths)

    for k, v in sorted(vars(args).items()):
        message = "\t"
        message +=  k.ljust(max_lenght, " ")
        message += " : {}".format(v)
        logging.info(message)

    logging.info("")


def convert_flot_to_int(value):

    if isinstance(value, float):
        value = int(value * 100)

    return value


def get_ranking_filename(dataset):
    filename = "{}/ranking_count_0{}.txt".format(PATH_RANKING, dataset)
    return filename


def get_original_filename(dataset, full=True):
    file_in = ""
    file_out = ""
    if full:
        file_in = "{}/{}".format(PATH_ORIGINAL, dataset)
        file_out = "{}.sort-k-3-3".format(file_in)
        cmd = "sort -k 3,3 {} > {}".format(file_in, file_out)
        logging.debug("running: {}".format(cmd))
        os.system(cmd)

    else:
        file_in = "{}".format(dataset)
        file_out = "{}.sort-k-3-3".format(file_in)

    return file_out


def get_original_unzip_filename(dataset, full=True):

    #print(files)
    #print(files[dataset-1])
    #sys.exit()

    file_out = ""
    if full:
        file_in = get_original_zip_filename(dataset, True)
        file_temp = "{}/S{}.temp".format(PATH_ORIGINAL, dataset)
        file_out = "{}/S{}.sort_u_1n_4n".format(PATH_ORIGINAL, dataset)
        if not os.path.isfile(file_out):
            cmd = "zcat {} > {}".format(file_in, file_temp)
            logging.info("running: {}".format(cmd))
            os.system(cmd)

            cmd = "./script_convert_trace_to_snapshot.sh -f {} -o {}".format(file_temp, file_out)
            run_cmd(cmd)
        else:
            logging.info("original_filename exists: {}".format(file_out))

    else:
        file_out = "S{}.sort_u_1n_4n".format(dataset)

    return file_out


def get_original_zip_filename(dataset, full=True):

    file = "{}".format(files[dataset])

    if full:
        file = "{}/{}".format(SOURCE_ZIP_FILES, file)

    return file


def get_training_filename(dataset, full=True):
    return get_original_filename(dataset, full)


def get_validation_swarm_file(file_in=DEFAULT_VALIDATION_DATASET):

    file_out = "{}.sort-k-3-3".format(file_in)
    cmd = "sort -k 3,3 {} > {}".format(file_in, file_out)
    logging.debug("running: {}".format(cmd))
    os.system(cmd)

    return file_out


def get_mon_failed_filename(dataset, mif, full=True):

    if mif is None:
        mif = 100 #all monitors

    filename = "S{}m{:0>2d}.sort_u_1n_4n".format(dataset, mif)
    if full:
        filename = "{}/{}".format(PATH_FAILED_MON, filename)
    return filename


def get_prob_failed_filename(dataset, pif, seed, full=True):

    pif = convert_flot_to_int(pif)
    filename = "{}.failed_pif-{:0>2d}_seed-{:0>3d}".format(get_original_filename(dataset, False), pif, seed)

    if full:
        filename = "{}/{}".format(PATH_FAILED_PROB, filename)

    return filename


def get_corrected_filename(dataset, mif, seed, threshold, window, full=True):
    if mif is None:
        mif = 100
    threshold = convert_flot_to_int(threshold)
    #filename = "{}.corrected_threshold-{:0>2d}_window-{:0>2d}".format(get_failed_filename(dataset, pif, seed, False), threshold, window)
    filename = "{}_mfi-{:0>2d}.corrected_threshold-{:0>2d}_window-{:0>2d}_epochs-{:0>4d}".format(get_original_unzip_filename(dataset, False),
                                                                      mif, threshold, window, NUM_EPOCHS)
    if full:
        filename = "{}/{}".format(PATH_CORRECTED, filename)

    return filename


def get_model_filename(training_file, number_blocks, window, trial, full=True):

    filename = "model_{}_number_blocks-{:0>2d}_window-{}_trial-{:0>2d}".format(
         training_file.split("/")[-1], number_blocks, window, trial)

    if full:
        filename = "{}/{}".format(PATH_MODEL, filename)

    return filename




# Custom argparse type representing a bounded int
# source: https://stackoverflow.com/questions/14117415/in-python-using-argparse-allow-only-positive-integers
class IntRange:

    def __init__(self, imin=None, imax=None):

        self.imin = imin
        self.imax = imax

    def __call__(self, arg):

        try:
            value = int(arg)

        except ValueError:
            raise self.exception()

        if (self.imin is not None and value < self.imin) or (self.imax is not None and value > self.imax):
            raise self.exception()

        return value

    def exception(self):

        if self.imin is not None and self.imax is not None:
            return argparse.ArgumentTypeError(f"Must be an integer in the range [{self.imin}, {self.imax}]")

        elif self.imin is not None:
            return argparse.ArgumentTypeError(f"Must be an integer >= {self.imin}")

        elif self.imax is not None:
            return argparse.ArgumentTypeError(f"Must be an integer <= {self.imax}")

        else:
            return argparse.ArgumentTypeError("Must be an integer")


def run_cmd(cmd):
    print(cmd)
    logging.info("")
    logging.info("Command line : {}".format(cmd))
    cmd_array = shlex.split(cmd)
    logging.debug("Command array: {}".format(cmd_array))
    if not args.demo:
        subprocess.run(cmd_array, check=True)


class Campaign():

    def __init__(self, datasets, pifs, number_blocks, thresholds, windows):
        self.datasets = datasets
        self.number_blocks = number_blocks
        self.thresholds = thresholds
        self.pifs = pifs
        self.windows = windows


def create_monitor_injected_fail_file(dataset, mif):
    '''
        -i      input file
        -o      output file
        -r      random number generator seed
        -p      failure probability - must be expressed between [0,100]

    :param dataset:
    :param pif:
    :param trial:
    :return:
    '''
    if mif is None:
        mif = 100 #all monitors

    cmd = "python3 emulate_monitor_failures.py "
    cmd += "--source {} ".format(get_original_zip_filename(dataset))
    cmd += "--output {} ".format(get_mon_failed_filename(dataset, mif))
    cmd += "--ranking {} ".format(get_ranking_filename(dataset))
    cmd += "--number {} ".format(mif)
    run_cmd(cmd)


def create_probability_injected_fail_file(dataset, pif, trial):
    '''
        -i      input file
        -o      output file
        -r      random number generator seed
        -p      failure probability - must be expressed between [0,100]

    :param dataset:
    :param pif:
    :param trial:
    :return:
    '''

    cmd = "./script_emulate_snapshot_failures.sh "
    cmd += "-i {} ".format(get_original_filename(dataset))
    cmd += "-o {} ".format(get_prob_failed_filename(dataset, pif, trial))
    cmd += "-r {} ".format(trial)
    cmd += "-p {} ".format(convert_flot_to_int(pif))
    run_cmd(cmd)


def check_files(files, error=False):
    internal_files = files
    if isinstance(files, str):
        internal_files = [files]

    for f in internal_files:
        if not os.path.isfile(f):
            if error:
                logging.info("ERROR: file not found! {}".format(f))
                sys.exit(1)
            else:
                logging.info("File not found! {}".format(f))
                return False
        else:
            logging.info(("File found: {}".format(f)))

    return True


def main():

    print("Creating the structure of directories...")

    for path in PATHS:

        cmd = "mkdir -p {}".format(path)
        print("path: {} cmd: {}".format(path, cmd))
        cmd_array = shlex.split(cmd)
        subprocess.run(cmd_array, check=True)

    print("done.")
    print("")

    parser = argparse.ArgumentParser(description='Torrent Trace Correct - Machine Learning')

    help_msg = 'append output logging file with analysis results (default={})'.format(DEFAULT_APPEND_OUTPUT_FILE)
    parser.add_argument("--append", "-a", default=DEFAULT_APPEND_OUTPUT_FILE, help=help_msg, action='store_true')

    help_msg = "demo mode (default={})".format(False)
    parser.add_argument("--demo", "-d", help=help_msg,  action='store_true')

    help_msg = "number of trials (default={})".format(DEFAULT_TRIALS)
    parser.add_argument("--trials", "-r", help=help_msg, default=DEFAULT_TRIALS, type=IntRange(1))

    help_msg = "start trials (default={})".format(DEFAULT_START_TRIALS)
    parser.add_argument("--start_trials", "-s", help=help_msg, default=DEFAULT_START_TRIALS, type=IntRange(0))

    help_msg = "Skip training of the machine learning model training?"
    parser.add_argument("--skip_train", "-t", default=False, help=help_msg, action='store_true')

    help_msg = "Campaign [demo, case, comparison] (default={})".format(DEFAULT_CAMPAIGN)
    parser.add_argument("--campaign", "-c", help=help_msg, default=DEFAULT_CAMPAIGN, type=str)

    help_msg = "verbosity logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
    parser.add_argument("--verbosity", "-v", help=help_msg, default=DEFAULT_VERBOSITY_LEVEL, type=int)

    global args
    args = parser.parse_args()

    logging_filename = '{}/run_tnsm_{}.log'.format(PATH_LOG, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    logging_format = '%(asctime)s\t***\t%(message)s'
    # configura o mecanismo de logging
    if args.verbosity == logging.DEBUG:
        # mostra mais detalhes
        logging_format = '%(asctime)s\t***\t%(levelname)s {%(module)s} [%(funcName)s] %(message)s'

    # formatter = logging.Formatter(logging_format, datefmt=TIME_FORMAT, level=args.verbosity)
    logging.basicConfig(format=logging_format, level=args.verbosity)

    # Add file rotating handler, with level DEBUG
    rotatingFileHandler = RotatingFileHandler(filename=logging_filename, maxBytes=100000, backupCount=5)
    rotatingFileHandler.setLevel(args.verbosity)
    rotatingFileHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(rotatingFileHandler)

    # imprime configurações para fins de log
    print_config(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1' #0
    #mifs = [20, 17, 16, 12, 11, 10, 9, 8, 7]
    mifs = [20, 17, 16, 12, 11, 10, 9, 8, 7]
    mifs = [7]
    c_demo = Campaign(datasets=[1], number_blocks=[32], thresholds=[.75], pifs=mifs, windows=[256])
    #
    # c_comparison = Campaign(datasets=[1], dense_layers=[3], thresholds=[.75], pifs=mifs, #7,11,17,10,16
    #                         rnas=["lstm_mode", "no-lstm_mode"], windows=[11])
    # c_case = Campaign(datasets=[3], dense_layers=[3], thresholds=[.75], pifs=[None], rnas=["lstm_mode"], windows=[11])
    #
    # if args.campaign == "demo":
    #     campaigns = [c_demo]
    # elif args.campaign == "case":
    #     campaigns = [c_case]
    # else:
    #     campaigns = [c_comparison]


    campaigns = [c_demo]
    logging.info("\n\n\n")
    logging.info("##########################################")
    logging.info(" TRAINING ")
    logging.info("##########################################")


    # 1
    input_dataset_training_in = 'dataset/training/failed_training/S1m07_20.sort_u_1n_4n'
    OUTPUT_DATASET_TRAINING_IN = 'samples_saved/samples_training_in/S1m07_20.sort_u_1n_4n'
    if not check_files("{}.npz".format(OUTPUT_DATASET_TRAINING_IN)):
        cmd = "python3 main.py CreateSamples"
        cmd += " --input_file_swarm {}".format(input_dataset_training_in)
        cmd += " --save_file_samples {}".format(OUTPUT_DATASET_TRAINING_IN)
        run_cmd(cmd)

    # 2
    INPUT_DATASET_TRAINING_OUT = 'dataset/training/original_training/S1m30_20.sort_u_1n_4n'
    OUTPUT_DATASET_TRAINING_OUT = 'samples_saved/samples_training_out/S1m30_20.sort_u_1n_4n'
    if not check_files("{}.npz".format(OUTPUT_DATASET_TRAINING_OUT)):
        cmd = "python3 main.py CreateSamples"
        cmd += " --input_file_swarm {}".format(INPUT_DATASET_TRAINING_OUT)
        cmd += " --save_file_samples {}".format(OUTPUT_DATASET_TRAINING_OUT)
        run_cmd(cmd)

    # 3
    INPUT_DATASET_PREDICT_IN = 'dataset/predict/S1m07_80.sort_u_1n_4n'
    OUTPUT_DATASET_PREDICT_OUT = 'samples_saved/samples_predict/S1m07_80.sort_u_1n_4n'
    if not check_files("{}.npz".format(OUTPUT_DATASET_PREDICT_OUT)):
        cmd = "python3 main.py CreateSamples"
        cmd += " --input_file_swarm {}".format(INPUT_DATASET_PREDICT_IN)
        cmd += " --save_file_samples {}".format(OUTPUT_DATASET_PREDICT_OUT)
        run_cmd(cmd)


    dense_layers_models = {}
    trials = range(args.start_trials, (args.start_trials + args.trials))
    time_start_campaign = datetime.datetime.now()
    training_dataset = "S1a"
    for trial in trials:
        for count_c, c in enumerate(campaigns):
            for number_blocks in c.number_blocks:
                for window in c.windows:

                    if not (number_blocks, window, trial) in dense_layers_models.keys():
                        logging.info("\tCampaign: {} number_blocks: {} Window: {}".format(count_c, number_blocks, window))

                        #check_files([training_file])

                        time_start_experiment = datetime.datetime.now()
                        logging.info(
                            "\t\t\t\t\t\t\t\tBegin: {}".format(time_start_experiment.strftime(TIME_FORMAT)))

                        model_filename = get_model_filename(OUTPUT_DATASET_TRAINING_IN, number_blocks, window, trial)
                        logging.debug("\tmodel_filename: {}".format(model_filename))
                        dense_layers_models[(number_blocks, window, trial)] = (model_filename)

                        if not args.skip_train:
                            cmd = "python3 main.py Training"
                            cmd += " --number_blocks {}".format(number_blocks)
                            cmd += " --window_width {}".format(window)
                            cmd += " --window_length {}".format(window)
                            cmd += " --epochs {}".format(NUM_EPOCHS)
                            cmd += " --load_samples_in {}".format(OUTPUT_DATASET_TRAINING_IN)
                            cmd += " --load_samples_out {}".format(OUTPUT_DATASET_TRAINING_OUT)
                            cmd += " --save_model {}".format(model_filename)
                            run_cmd(cmd)

                        time_end_experiment = datetime.datetime.now()
                        logging.info("\t\t\t\t\t\t\tEnd                : {}".format(
                            time_end_experiment.strftime(TIME_FORMAT)))
                        logging.info("\t\t\t\t\t\t\tExperiment duration: {}".format(
                            time_end_experiment - time_start_experiment))

                        check_files(["{}.h5".format(model_filename), "{}.json".format(model_filename)])


    logging.info("\n\n\n")
    logging.info("##########################################")
    logging.info(" EVALUATION ")
    logging.info("##########################################")

    count_trial = 1
    for trial in trials:
        logging.info("\tTrial {}/{} ".format(count_trial, len(trials)))
        count_trial += 1
        count_campaign = 1
        for c in campaigns:
            logging.info("\t\tCampaign {}/{} ".format(count_campaign, len(campaigns)))
            count_campaign += 1
            count_dataset = 1
            for dataset in c.datasets:
                logging.info("\t\t\tDatasets {}/{} ".format(count_dataset, len(c.datasets)))
                count_dataset += 1

                original_swarm_file = get_original_unzip_filename(dataset, 30) #30 monitores -> sem falha
                logging.info("\t\t\t#original_swarm_file: {}".format(original_swarm_file))

                count_number_blocks = 1
                for number_blocks in c.number_blocks:
                    logging.info("\t\t\t\tnumber_blocks {}/{} ".format(count_number_blocks, len(c.number_blocks)))
                    count_number_blocks += 1
                    count_pif = 1
                    for pif in c.pifs:
                        logging.info("\t\t\t\t\tPifs {}/{} ".format(count_pif, len(c.pifs)))
                        count_pif += 1

                        failed_swarm_file = get_mon_failed_filename(dataset, pif)
                        logging.info("\t\t\t#failed_swarm_file: {}".format(failed_swarm_file))

                        count_threshold = 1
                        for threshold in c.thresholds:
                            logging.info("\t\t\t\t\t\t\tThreshold {}/{} ".format(count_threshold, len(c.thresholds)))
                            count_threshold += 1

                            count_window = 1
                            for window in c.windows:
                                logging.info("\t\t\t\t\t\t\t\tWindow {}/{} ".format(count_window, len(c.windows)))
                                count_window += 1

                                (model_filename) = dense_layers_models[(number_blocks, window, trial)]

                                corrected_swarm_file = get_corrected_filename(dataset, pif, trial, threshold, window)
                                time_start_experiment = datetime.datetime.now()
                                logging.info("\t\t\t\t\t\t\t\tBegin: {}".format(time_start_experiment.strftime(TIME_FORMAT)))

                                if not args.skip_train:
                                    check_files(["{}.h5".format(model_filename), "{}.json".format(model_filename)])

                                check_files([original_swarm_file, failed_swarm_file])
                                print("TESTE original_swarm_file")
                                if not check_files(["{}.npz".format(original_swarm_file)]):
                                    cmd = "python3 main.py CreateSamples"
                                    cmd += " --input_file_swarm {}".format(original_swarm_file)
                                    cmd += " --save_file_samples {}".format(original_swarm_file)

                                print("TESTE failed_swarm_file")
                                if not check_files(["{}.npz".format(failed_swarm_file)]):
                                    cmd = "python3 main.py CreateSamples"
                                    cmd += " --input_file_swarm {}".format(failed_swarm_file)
                                    cmd += " --save_file_samples {}".format(failed_swarm_file)

                                cmd = "python3 main.py Predict"
                                cmd += " --input_predict {}".format(failed_swarm_file)
                                cmd += " --output_predict {}".format(corrected_swarm_file)
                                cmd += " --load_model {}".format(model_filename)
                                run_cmd(cmd)

                                RESULT_METRICS = 'results/results_tnsm_mif.txt'
                                cmd = "python3 main.py Analyse"
                                cmd += " --file_original {}".format(original_swarm_file)
                                cmd += " --file_corrected {}".format(corrected_swarm_file)
                                cmd += " --file_failed {}".format(failed_swarm_file)
                                cmd += " --file_analyse {}".format(RESULT_METRICS)
                                run_cmd(cmd)

                                time_end_experiment = datetime.datetime.now()
                                logging.info("\t\t\t\t\t\t\t\tEnd                : {}".format(time_end_experiment.strftime(TIME_FORMAT)))
                                logging.info("\t\t\t\t\t\t\t\tExperiment duration: {}".format(time_end_experiment - time_start_experiment))

                                check_files([corrected_swarm_file])

    time_end_campaign = datetime.datetime.now()
    logging.info("\t Campaign duration: {}".format(time_end_campaign - time_start_campaign))


if __name__ == '__main__':
    sys.exit(main())

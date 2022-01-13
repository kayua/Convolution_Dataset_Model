import logging

from sklearn.metrics import recall_score, f1_score, roc_curve, accuracy_score, precision_score, confusion_matrix
from tqdm import tqdm
import datetime
DEFAULT_PATH_ANALYSES = 'analyses/'
DEFAULT_PATH_LOG = 'logs/'


class Analyse:

    def __init__(self, args):

        self.snapshot_column_position = args.snapshot_column_position
        self.peer_column_position = args.peer_column_position


    def load_swarm(self, file_swarm):

        file_pointer_swarm = open(file_swarm, 'r')
        line_swarm_file = file_pointer_swarm.readlines()

        for i, swarm_line in enumerate(line_swarm_file):

            swarm_line_in_list = swarm_line.split(' ')
            snapshot_value = int(swarm_line_in_list[self.snapshot_column_position - 1])
            peer_value = int(swarm_line_in_list[self.peer_column_position - 1])


    @staticmethod
    def get_accuracy_score(y_true, y_predicted):

        score_accuracy = accuracy_score(y_true, y_predicted)
        logging.debug('Predict accuracy score: {}'.format(score_accuracy))
        return score_accuracy

    @staticmethod
    def get_precision_score(y_true, y_predicted):

        score_precision = precision_score(y_true, y_predicted, average='macro')
        logging.debug('Predict precision score: {}'.format(score_precision))
        return score_precision

    @staticmethod
    def get_recall_score(y_true, y_predicted):

        score_recall = recall_score(y_true, y_predicted, average='macro')
        logging.debug('Predict recall score: {}'.format(score_recall))
        return score_recall

    @staticmethod
    def get_f1_score(y_true, y_predicted):

        score_f1 = f1_score(y_true, y_predicted, average='macro')
        logging.debug('Predict F1 score: {}'.format(score_f1))
        return score_f1

















    def write_results_analyse(self, start_time, size_window_left, size_window_right):

        analyse_results = open(self.analyse_file, 'w')

        analyse_results.write('\nBEGIN ############################################\n\n')
        analyse_results.write(' RESULTS \n')
    
        analyse_results.write('  Size files:           \n')
        analyse_results.write('-----------------------------\n')
        analyse_results.write('  Total Traces original file         : {}\n'.format(self.size_list_original))
        analyse_results.write('  Total Traces failed file           : {}\n'.format(self.size_list_failed))
        analyse_results.write('  Total Traces corrected file        : {}\n'.format(self.size_list_corrected))

        faults = self.size_list_original - self.size_list_failed
        analyse_results.write('  Fails (Original-failed)            : {}\n'.format(faults))

        modification = self.size_list_corrected - self.size_list_failed
        analyse_results.write('  Modifications (Original-corrected) : {}\n'.format(modification))

        analyse_results.write('------------------------------\n')
        analyse_results.write('            Analyse:          \n')
        analyse_results.write('------------------------------\n')
        analyse_results.write('  Found in [Original, Corrected, Failed]: {}\n'.format(self.trace_found_in_original_and_failed_and_corrected))
        analyse_results.write('  Found in [Original, Corrected]        : {}\n'.format(self.trace_found_in_original_and_corrected))
        analyse_results.write('  Found in [Original, Failed]           : {}\n'.format(self.trace_found_in_original_and_failed))
        analyse_results.write('------------------------------\n')
        analyse_results.write('            Scores:           \n')
        analyse_results.write('------------------------------\n')
        tp = self.trace_found_in_original_and_corrected-self.trace_found_in_original_and_failed
        analyse_results.write('  True positive  (TP): {}\n'.format(tp))
        fp = self.size_list_corrected-self.trace_found_in_original_and_corrected
        analyse_results.write('  False positive (FP): {}\n'.format(fp))
        fn = self.size_list_original-self.trace_found_in_original_and_corrected
        analyse_results.write('  False negative (FN): {}\n'.format(fn))

        tn = self.size_list_original -tp -(fp+fn)
        analyse_results.write('  True negative  (TN): {}\n'.format(tn))

        line ="#SUMMARY#"
        line += ";{}".format(size_window_left+size_window_right+1)
        line += ";{}".format(self.size_list_original)
        line += ";{}".format(faults)
        line += ";{}".format(modification)
        line += ";{}".format(tp)
        line += ";{}".format(fp)
        line += ";{}".format(fn)
        line += ";{}".format(tn)
        line += "\n"
        line = line.replace(";", "\t")
        print(line)

        analyse_results.write(line)
        analyse_results.write('\nEND ############################################\n\n')
        analyse_results.write('\n\n\n')
        analyse_results.close()

import logging
from datetime import datetime

from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


DEFAULT_PATH_ANALYSES = 'analyses/'
DEFAULT_PATH_LOG = 'logs/'


class Analyse:

    def __init__(self, args):

        self.snapshot_column_position = args.snapshot_column_position
        self.peer_column_position = args.peer_column_position
        self.corrected_swarm_file = args.corrected_swarm_file
        self.failed_swarm_file = args.failed_swarm_file
        self.original_swarm_file = args.original_swarm_file
        self.topology = args.topology
        self.analyse_file_mode = args.analyse_file_mode
        self.analyse_file_results = args.analyse_file_results

        self.trace_found_in_original_and_failed = 0
        self.trace_found_in_original_and_corrected = 0
        self.trace_found_in_original_and_failed_and_corrected = 0
        self.number_original_swarm_lines = 0
        self.number_predicted_swarm_lines = 0
        self.number_failed_swarm_lines = 0
        self.threshold = 1
        self.seed = 0
        self.pif = 0
        self.dictionary_original_swarm = {}
        self.dictionary_predicted_swarm = {}
        self.dictionary_failed_swarm = {}

        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

    @staticmethod
    def add_value_to_dic(dictionary, snapshot_id, peer_id):

        try:

            snapshot = dictionary[snapshot_id]
            snapshot[peer_id] = True
            dictionary[snapshot_id] = snapshot

        except KeyError:
            new_dic = {peer_id: True}
            dictionary[snapshot_id] = new_dic

        return dictionary

    def load_swarm(self, file_swarm):

        file_pointer_swarm = open(file_swarm, 'r')
        line_swarm_file = file_pointer_swarm.readlines()
        temporary_dictionary = {}
        number_true_peers = 0

        for i, swarm_line in enumerate(line_swarm_file):

            swarm_line_in_list = swarm_line.split(' ')
            snapshot_value = int(swarm_line_in_list[self.snapshot_column_position - 1])
            peer_value = int(swarm_line_in_list[self.peer_column_position - 1])
            temporary_dictionary = self.add_value_to_dic(temporary_dictionary, snapshot_value, peer_value)
            number_true_peers += 1


        return temporary_dictionary, number_true_peers

    def comparison_of_results(self, snapshot_id, peer_id):


        if self.dictionary_failed_swarm[snapshot_id][peer_id]:

            self.trace_found_in_original_and_failed += 1

        if self.dictionary_predicted_swarm[snapshot_id][peer_id]:

            self.trace_found_in_original_and_corrected += 1

        if self.dictionary_failed_swarm[snapshot_id][peer_id]:

            if self.dictionary_predicted_swarm[snapshot_id][peer_id]:

                self.trace_found_in_original_and_failed_and_corrected += 1

    def get_all_metrics(self):

        self.dictionary_original_swarm, self.number_original_swarm_lines = self.load_swarm(self.original_swarm_file)
        self.dictionary_predicted_swarm,  self.number_predicted_swarm_lines = self.load_swarm(self.corrected_swarm_file)
        self.dictionary_failed_swarm, self.number_failed_swarm_lines = self.load_swarm(self.failed_swarm_file)

        for key_first_dic, value_first_dic in self.dictionary_original_swarm.items():

            for key_second_dic, value_second_dic in value_first_dic.items():

                self.comparison_of_results(key_first_dic, key_second_dic)

        self.true_positives = self.trace_found_in_original_and_corrected - self.trace_found_in_original_and_failed
        self.false_positives = self.number_original_swarm_lines - self.trace_found_in_original_and_corrected
        self.false_negatives = self.number_original_swarm_lines - self.trace_found_in_original_and_corrected
        false_positives_false_negatives = (self.false_positives + self.false_negatives)
        self.true_negatives = self.number_original_swarm_lines - self.true_positives - false_positives_false_negatives

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

    def write_results_analyse(self):

        analyse_results = open(self.analyse_file_results, self.analyse_file_mode)


        analyse_results.write('\nBEGIN ############################################\n\n')
        analyse_results.write(' RESULTS \n')
        analyse_results.write("  Now      : {}\n".format(datetime.now()))
        analyse_results.write("  Topology : {}\n".format(self.topology))
        analyse_results.write("  Threshold: {}\n".format(self.threshold))
        analyse_results.write("  PIF      : {}%\n".format(int(self.pif * 100)))
        analyse_results.write("  Dataset  : {}\n".format(self.original_swarm_file))
        analyse_results.write("  Seed     : {}\n\n".format(self.seed))

        analyse_results.write('  Size files:           \n')
        analyse_results.write('-----------------------------\n')
        analyse_results.write('  Total Traces original file         : {}\n'.format(self.number_original_swarm_lines))
        analyse_results.write('  Total Traces failed file           : {}\n'.format(self.number_failed_swarm_lines))
        analyse_results.write('  Total Traces corrected file        : {}\n'.format(self.number_predicted_swarm_lines))

        falhas = self.number_original_swarm_lines - self.number_failed_swarm_lines
        analyse_results.write('  Fails (Original-failed)            : {}\n'.format(falhas))

        modificacoes = self.number_predicted_swarm_lines - self.number_failed_swarm_lines
        analyse_results.write('  Modifications (Original-corrected) : {}\n'.format(modificacoes))

        analyse_results.write('------------------------------\n')
        analyse_results.write('            Analyse:          \n')
        analyse_results.write('------------------------------\n')
        analyse_results.write('  Found in [Original, Corrected, Failed]: {}\n'.format(
                self.trace_found_in_original_and_failed_and_corrected))
        analyse_results.write(
                '  Found in [Original, Corrected]        : {}\n'.format(self.trace_found_in_original_and_corrected))
        analyse_results.write(
                '  Found in [Original, Failed]           : {}\n'.format(self.trace_found_in_original_and_failed))
        analyse_results.write('------------------------------\n')
        analyse_results.write('            Scores:           \n')
        analyse_results.write('------------------------------\n')
        analyse_results.write('  True positive  (TP): {}\n'.format(self.true_positives))
        analyse_results.write('  False positive (FP): {}\n'.format(self.false_positives))
        analyse_results.write('  False negative (FN): {}\n'.format(self.false_negatives))
        analyse_results.write('  True negative  (TN): {}\n'.format(self.true_negatives))


        linha = "#SUMMARY#"
        linha += ";{}".format(self.topology)
        linha += ";{}".format(self.number_original_swarm_lines)
        linha += ";{}".format(falhas)
        linha += ";{}".format(self.threshold)
        linha += ";{}%".format(int(self.pif * 100))
        linha += ";{}".format(self.original_swarm_file)
        linha += ";{}".format(self.threshold)
        linha += ";{}".format(self.seed)
        linha += ";{}".format(self.true_positives)
        linha += ";{}".format(self.false_positives)
        linha += ";{}".format(self.true_negatives)
        linha += ";{}".format(self.false_negatives)
        linha += "\n"
        print(linha)

        linha = "#SUMNEW#"
        linha += ";{}".format(self.topology)
        linha += ";{}".format(self.threshold)

        linha += ";{}%".format(int(self.pif * 100))
        linha += ";{}".format(self.original_swarm_file)
        linha += ";{}".format(self.seed)

        linha += ";{}".format(self.number_original_swarm_lines)
        linha += ";{}".format(falhas)
        linha += ";{}".format(modificacoes)

        linha += ";{}".format(self.true_positives)
        linha += ";{}".format(self.false_positives)
        linha += ";{}".format(self.true_negatives)
        linha += ";{}".format(self.false_negatives)
        linha += "\n"
        print(linha)
        analyse_results.write(linha)

        analyse_results.write('\nEND ############################################\n\n')
        analyse_results.write('\n\n\n')
        analyse_results.close()

a = Analyse()
a.get_all_metrics()




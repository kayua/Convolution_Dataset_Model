from subprocess import Popen, PIPE
import numpy


class Dataset:

    def __init__(self):

        self.snapshot_column_position = 1
        self.peer_column_position = 2
        self.feature_window_length = 256
        self.feature_window_width = 256
        self.break_point = 1
        self.matrix_features = []
        self.number_block_per_samples = 32
        self.input_file_swarm_sorted = 'S4'
        self.features = []
        self.input_feature = []
        self.feature_input = []
        self.snapshot_id = self.feature_window_length

    def allocation_matrix(self):

        size_matrix_allocation_width = self.feature_window_width * self.number_block_per_samples

        for i in range(len(self.matrix_features), size_matrix_allocation_width):

            self.matrix_features.append([0 for _ in range(self.feature_window_length)])

    def clean_matrix(self):

        for i in range(len(self.matrix_features)):

            for j in range(len(self.matrix_features[0])):

                self.matrix_features[i][j] = 0

    def insert_in_matrix(self, snapshot_id, peer_id):

        if snapshot_id > self.snapshot_id:

            self.snapshot_id = self.snapshot_id + self.feature_window_length
            self.feature_input.append(numpy.array(self.matrix_features))
            self.clean_matrix()

        if (snapshot_id % self.feature_window_length) != 0:

            self.matrix_features[peer_id][(snapshot_id % self.feature_window_length)-1] = 1

        else:

            self.matrix_features[peer_id][self.feature_window_length-1] = 1

    def load_swarm_to_feature(self):

        self.allocation_matrix()
        file_pointer_swarm = open(self.input_file_swarm_sorted, 'r')
        line_swarm_file = file_pointer_swarm.readlines()

        for i, swarm_line in enumerate(line_swarm_file):

            swarm_line_in_list = swarm_line.split(' ')
            snapshot_value = int(swarm_line_in_list[self.snapshot_column_position - 1])
            peer_value = int(swarm_line_in_list[self.peer_column_position - 1])
            self.insert_in_matrix(snapshot_value, peer_value)

        self.feature_input.append(numpy.array(self.matrix_features))
        self.clean_matrix()
        self.cut_features()
        self.cast_all_features_to_swarm()


    def cut_features(self):

        results = []

        for i in self.feature_input:

            for j in range(0, self.feature_window_width*self.number_block_per_samples, self.feature_window_width):

                results.append(i[j:j+self.feature_window_width])

        self.features = results

    @staticmethod
    def restore_matrix(matrix):

        matrix_results = []

        for i in matrix:

            matrix_results.extend(i)
        return matrix_results

    def cast_feature_to_swarm(self, feature, position, pointer):

        results = self.restore_matrix(feature)

        for i in range(len(results)):

            for j in range(len(results[0])):

                if results[i][j]:

                    pointer.write('{} {}\n'.format(j+position+1, i))

    def cast_all_features_to_swarm(self):

        output = open('output_saida.txt', 'w')
        for i, j in enumerate(range(0, len(self.features), self.number_block_per_samples)):
            self.cast_feature_to_swarm(self.features[j:j+self.number_block_per_samples], i*self.feature_window_length, output)

    def sort(self):

        sequence_commands = 'sort -n -k{},{} '.format(self.snapshot_column_position, self.snapshot_column_position)
        sequence_commands += '-k{},{} '.format(self.peer_column_position, self.peer_column_position)
        sequence_commands += '{} -o {}'.format('output_saida.txt', 'saida_sorted.txt')
        external_process = Popen(sequence_commands.split(' '), stdout=PIPE, stderr=PIPE)
        external_process.communicate()


a = Dataset()
a.load_swarm_to_feature()
a.
a.
a.sort()
exit()
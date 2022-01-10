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
        self.list_features = []
        self.feature_input = None
        self.feature_output = None

    def allocation_matrix(self):

        if self.feature_window_length % 2:
            print('Erro: Use um numero de base binaria para de comprimento')
            exit(-1)

        if self.number_block_per_samples % 2:
            print('Erro: Use um numero de base binaria para de blocos')
            exit(-1)

        if self.feature_window_width % 2:
            print('Erro: Use um numero de base binaria para de largura')
            exit(-1)

        size_matrix_allocation_width = self.feature_window_width * self.number_block_per_samples

        for i in range(len(self.matrix_features), size_matrix_allocation_width):

            self.matrix_features.append([0 for _ in range(self.feature_window_length)])

    def clean_matrix(self):

        for i in range(len(self.matrix_features)):

            for j in range(len(self.matrix_features[i])):

                self.matrix_features[i][j] = 0

    def create_feature(self):

        for i in range(self.number_block_per_samples):

            start_feature = i * self.feature_window_width
            end_feature = (i + 1) * self.feature_window_width
            feature_matrix = numpy.array(self.matrix_features[start_feature: end_feature])
            self.list_features.append(feature_matrix)

    def add_peer_in_matrix(self, snapshot, peer_id):

        self.matrix_features[peer_id][(snapshot % self.feature_window_length) - 1] = 1

    def cast_list_features_to_numpy(self):

        self.feature_input = numpy.array(self.list_features, dtype=numpy.float32)
        self.list_features = None

    def load_swarm_to_feature(self):

        self.allocation_matrix()
        file_pointer_swarm = open(self.input_file_swarm_sorted, 'r')
        line_swarm_file = file_pointer_swarm.readlines()

        for _, swarm_line in enumerate(line_swarm_file):

            swarm_line_in_list = swarm_line.split(' ')
            snapshot_value = int(swarm_line_in_list[self.snapshot_column_position - 1])
            peer_value = int(swarm_line_in_list[self.peer_column_position - 1])


            if (snapshot_value % self.feature_window_length == 1) and snapshot_value != self.break_point:

                self.break_point = snapshot_value
                self.create_feature()
                self.clean_matrix()
                self.add_peer_in_matrix(snapshot_value, peer_value)
            else:

                self.add_peer_in_matrix(snapshot_value, peer_value)


        self.create_feature()
        self.clean_matrix()
        self.cast_list_features_to_numpy()

    @staticmethod
    def get_matrix(list_matrix):

        results = []

        for i in list_matrix:
            results.extend(i.T)

        return numpy.array(results)


    def write_file(self, matrix, pointer, file):

        for i in range(self.feature_window_width):

            for j in range(self.feature_window_length):

                if matrix[i][j] > 0.8:

                    file.write('{} {}\n'.format(i+pointer+1, j))



    def cast_matrix_to_swarm(self):

        pointer_file_swarm = open('S4_output.txt', 'w')
        for i in range(0, len(self.feature_input), self.number_block_per_samples):
            c = self.get_matrix(self.feature_input[i:i+self.number_block_per_samples])
            self.write_file(c.tolist(), i, pointer_file_swarm)








a = Dataset()
a.load_swarm_to_feature()
a.cast_matrix_to_swarm()

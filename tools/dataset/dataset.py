import numpy


class Dataset:

    def __init__(self):

        self.snapshot_column_position = 1
        self.peer_column_position = 2
        self.feature_window_length = 12
        self.feature_window_width = 32
        self.break_point = 1
        self.matrix_features = []
        self.number_block_per_samples = 2
        self.input_file_swarm_sorted = 'S4_old'
        self.list_features = []
        self.feature_input = []
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

            for j in range(len(self.matrix_features[0])):

                self.matrix_features[i][j] = 0

    def insert_in_matrix(self, snapshot_id, peer_id):

        self.matrix_features[peer_id][(snapshot_id % self.feature_window_length)-1] = 1

    def create_samples(self):

        results = []

        for i in range(0, self.feature_window_width*self.number_block_per_samples, self.feature_window_width):

            results.append(self.matrix_features[i:i+self.feature_window_width])









    def load_swarm_to_feature(self):

        self.allocation_matrix()
        file_pointer_swarm = open(self.input_file_swarm_sorted, 'r')
        line_swarm_file = file_pointer_swarm.readlines()
        last_snapshot_read = 0

        for _, swarm_line in enumerate(line_swarm_file):

            swarm_line_in_list = swarm_line.split(' ')
            snapshot_value = int(swarm_line_in_list[self.snapshot_column_position - 1])
            peer_value = int(swarm_line_in_list[self.peer_column_position - 1])

            if (snapshot_value % self.feature_window_length == 0) and last_snapshot_read != 0:

                last_snapshot_read = snapshot_value
                self.insert_in_matrix(snapshot_value, peer_value)
                self.create_samples()
                self.clean_matrix()

            else:

                self.insert_in_matrix(snapshot_value, peer_value)

        self.create_samples()
        self.clean_matrix()

    def show_matrix(self):

        for i in range(len(self.matrix_features)):

            print(self.matrix_features[i])

            print('\n')






a = Dataset()
a.load_swarm_to_feature()
a.show_matrix()

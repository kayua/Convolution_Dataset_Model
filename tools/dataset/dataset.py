import numpy


class Dataset:

    def __init__(self):

        self.snapshot_column_position = 1
        self.peer_column_position = 2
        self.feature_window_length = 4
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
            #exit(-1)

        if self.number_block_per_samples % 2:
            print('Erro: Use um numero de base binaria para de blocos')
            #exit(-1)

        if self.feature_window_width % 2:
            print('Erro: Use um numero de base binaria para de largura')
            #exit(-1)

        size_matrix_allocation_width = self.feature_window_width * self.number_block_per_samples

        for i in range(len(self.matrix_features), size_matrix_allocation_width):

            self.matrix_features.append([0 for _ in range(self.feature_window_length)])

    def clean_matrix(self):

        for i in range(len(self.matrix_features)):

            for j in range(len(self.matrix_features[0])):

                self.matrix_features[i][j] = 0

    def insert_in_matrix(self, snapshot_id, peer_id):

        if (snapshot_id % self.feature_window_length) != 0:
            self.matrix_features[peer_id][(snapshot_id % self.feature_window_length)-1] = 1

        else:
            self.matrix_features[peer_id][self.feature_window_length-1] = 1

    def create_samples(self):

        results = []

        for i in range(0, self.feature_window_width*self.number_block_per_samples, self.feature_window_width):

            results.append(numpy.array(self.matrix_features[i:i+self.feature_window_width]))

        self.feature_input.extend(numpy.array(results))

    def load_swarm_to_feature(self):

        self.allocation_matrix()
        file_pointer_swarm = open(self.input_file_swarm_sorted, 'r')
        line_swarm_file = file_pointer_swarm.readlines()
        break_point = 0


        for i, swarm_line in enumerate(line_swarm_file):

            swarm_line_in_list = swarm_line.split(' ')
            snapshot_value = int(swarm_line_in_list[self.snapshot_column_position - 1])
            peer_value = int(swarm_line_in_list[self.peer_column_position - 1])
            if i == 0:
                break_point = snapshot_value

            if ((snapshot_value % self.feature_window_length) == 1) and break_point != snapshot_value:


                self.create_samples()
                self.clean_matrix()
                self.insert_in_matrix(snapshot_value, peer_value)
                break_point = snapshot_value

            else:

                self.insert_in_matrix(snapshot_value, peer_value)

    @staticmethod
    def concatenate(list_matrix):

        results = []

        for i in list_matrix:

            results.extend(i)

        return numpy.array(results)

    def cast_feature_to_swarm(self):

        ouput = open('saida.txt', 'w')
        result = []
        for i in range(0, len(self.feature_input), self.number_block_per_samples):
            result.append(self.concatenate(self.feature_input[i:i+self.number_block_per_samples]))

        x = numpy.array(result)

        for i in range(len(x)):

            for j in range(len(x[0])):

                for l in range(len(x[0][0])):

                    ouput.write('{} {}\n'.format(j, l+(i*)))

                print('\n')

            print('\n\n')

        exit()

    def show_matrix(self):


        for i in range(len(self.feature_input)):

            for j in range(len(self.feature_input[0])):

                print(self.feature_input[i][j])

            print('\n')






a = Dataset()
a.load_swarm_to_feature()
a.cast_feature_to_swarm()

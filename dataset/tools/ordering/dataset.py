

class Dataset:

    def __init__(self, args):

        self.feature_window_width = args.width_window
        self.feature_window_length = args.length_window
        self.matrix_features = []
        self.number_block_per_samples = 8

    def allocation_matrix(self, length):

        for i in range(self.number_block_per_samples):

            for j in range(len(self.matrix_features), length + 1):

                self.matrix_features.append([0 for x in range(self.feature_window_width)])





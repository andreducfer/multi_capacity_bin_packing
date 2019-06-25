import numpy as np


class Instance:
    def __init__(self, instance_path, solution_path, dataset_name, seed, max_time):
        self.instance_path = instance_path
        self.solution_path = solution_path
        self.dataset_name = dataset_name
        self.seed = seed
        self.max_time = max_time

        self._load_dataset(self.instance_path)

    def _load_dataset(self, instance_path):
        with open(instance_path) as fp:
            self.data_type = fp.readline().split()[0]
            self.opt_solution = int(fp.readline().split()[0])
            self.num_samples = int(fp.readline().split()[0])
            self.bin_constraints = np.array(fp.readline().split()).astype(np.float)
            self.num_constraints = int(len(self.bin_constraints))
            self.data_values = np.zeros((self.num_samples, self.num_constraints), dtype=np.float)

            for i, line in enumerate(fp):
                values = line.split()
                self.data_values[i] = np.array(values[:]).astype(np.float)

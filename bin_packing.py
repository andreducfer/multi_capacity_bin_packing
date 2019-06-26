from data_handler import Instance

import numpy as np


class Bin:
    def __init__(self, instance, samples_id=None):
        if not isinstance(instance, Instance):
            raise TypeError("Error: instance variable is not data_handler.Instance. Type: " + type(instance))

        self.instance = instance                            # Access to the problem and dataset parameters
        self.used_space = np.zeros((1,2), dtype=float)      # Used space by samples inside the bin
        self.fitness = 0

        if samples_id is None:
            self.samples = []                               # Samples in this bin
        else:
            for sample_id in samples_id:
                self.add_sample(sample_id)


    def evaluate(self):
        fitness = (self.used_space[0][0] / self.instance.bin_constraints[0] + self.used_space[0][1] / self.instance.bin_constraints[1]) ** 2
        self.fitness = fitness


    def verify_capacity(self, sample_id):
        partial_used_space = np.copy(self.used_space)
        partial_used_space += self.instance.data_values[sample_id]

        if self.instance.bin_constraints[0] >= partial_used_space[0][0] and self.instance.bin_constraints[1] >= partial_used_space[0][1]:
            return True

        return False


    def add_sample(self, sample_id):
        self.samples.append(sample_id)
        self.used_space += self.instance.data_values[sample_id]


class Solution:
    def __init__(self, instance):
        if not isinstance(instance, Instance):
            raise TypeError("Error: instance variable is not data_handler.Instance. Type: " + type(instance))

        self.instance = instance
        self.bin_packs = []
        self.eliminated_elements = []
        self.size_population = 2
        self.population = []


    def get_num_bins(self):
        return len(self.bin_packs)


class Greedy:
    def __init__(self, instance, solution):
        if not isinstance(instance, Instance):
            raise TypeError("Error: instance variable is not data_handler.Instance. Type: " + type(instance))
        if not isinstance(solution, Solution):
            raise TypeError("ERROR: solution variable is not decision_tree.Solution. Type: " + type(solution))
        self.instance = instance
        self.solution = solution


    def construct_population(self):
        for i in range(self.solution.size_population):
            self.greedy_construction()
            self.solution.population.append(self.solution.bin_packs)

        self.crossover(self.solution.population[0], self.solution.population[1])


    def greedy_construction(self, index_to_start_reconstruction=None):
        # If this index is not None, we need to destroy the solution starting at this index
        if index_to_start_reconstruction != None:
            del self.solution.bin_packs[index_to_start_reconstruction:]
        else:
            self.solution.bin_packs = []

        # List of index to sort matrix of data by first column and sort descending
        index_to_sort_descending = np.lexsort((-self.instance.data_values[:, 1], -self.instance.data_values[:, 0]))
        self.instance.data_values = self.instance.data_values[index_to_sort_descending]

        data_indexes = list(range(len(self.instance.data_values)))

        while len(data_indexes) > 0:
            new_bin = self._create_and_fill_bin(data_indexes)

            self.solution.bin_packs.append(new_bin)

            data_indexes_to_new_bin = new_bin.samples

            for data_index_to_new_bin in data_indexes_to_new_bin:
                data_indexes.remove(data_index_to_new_bin)


    def _create_and_fill_bin(self, data_indexes):
        new_bin = Bin(self.instance)
        for data_index in data_indexes:
            if new_bin.verify_capacity(data_index):
                new_bin.add_sample(data_index)

        new_bin.evaluate()
        return new_bin


    def crossover(self, first_parent, second_parent):
        copy_first_parent = np.copy(first_parent)
        copy_second_parent = np.copy(second_parent)

        # Generate random numbers with at most the minor size of the parents and use to split the two parents
        min_size_index_to_split = len(copy_first_parent) if len(copy_first_parent) < len(copy_second_parent) else len(copy_second_parent)
        random_numbers = np.random.randint(min_size_index_to_split, size=4)

        # Get 2 random numbers to use to split the first parent and sort
        first_parent_splits = random_numbers[:2]
        index_sorted_first_parent = np.argsort(first_parent_splits)
        first_parent_splits = first_parent_splits[index_sorted_first_parent]

        # Get 2 random numbers to use to split the second parent and sort
        second_parent_splits = random_numbers[2:]
        index_sorted_second_parent = np.argsort(second_parent_splits)
        second_parent_splits = second_parent_splits[index_sorted_second_parent]

        # Elements of parents to insert in oposite child
        bins_splited_first_child = np.array(copy_first_parent[first_parent_splits[0]:first_parent_splits[1]])
        bins_splited_second_child = np.array(copy_second_parent[second_parent_splits[0]:second_parent_splits[1]])

        # Create first child
        first_child = np.concatenate([copy_first_parent[:first_parent_splits[0]],
                                     bins_splited_second_child,
                                     copy_first_parent[first_parent_splits[0]:]])

        # Create second child
        second_child = np.concatenate([copy_second_parent[:second_parent_splits[0]],
                                       bins_splited_first_child,
                                       copy_second_parent[second_parent_splits[0]:]])

        self._delete_bins_with_duplicated_elements(bins_splited_first_child)
        self._delete_bins_with_duplicated_elements(bins_splited_second_child)


    def _delete_bins_with_duplicated_elements(self, new_bins):
        # Find bins with repeated elements
        bin_with_equal_elements = []
        for new_bin in new_bins:
            for new_element in new_bin.samples:
                equal_element = False
                for current_bin in self.solution.bin_packs:
                    for current_element in current_bin.samples:
                        if current_element == new_element:
                            equal_element = True
                            break
                    if equal_element == True:
                        bin_with_equal_elements.append(current_bin)
                        for element_to_eliminate in current_bin.samples:
                            self.solution.eliminated_elements.append(element_to_eliminate)
                        break
                if equal_element == True:
                    break

        # Remove bins with repeated elements
        for bin in bin_with_equal_elements:
            self.solution.bin_packs.remove(bin)
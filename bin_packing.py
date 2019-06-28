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


    def add_sample(self, sample_ids):
        for sample_id in sample_ids:
            self.samples.append(sample_id)
            self.used_space += self.instance.data_values[sample_id]


    def remove_sample(self, sample_ids):
        for sample_id in sample_ids:
            self.samples.remove(sample_id)
            self.used_space -= self.instance.data_values[sample_id]


class Solution:
    def __init__(self, instance):
        if not isinstance(instance, Instance):
            raise TypeError("Error: instance variable is not data_handler.Instance. Type: " + type(instance))

        self.instance = instance
        self.bin_packs = []
        self.eliminated_elements = []
        self.size_population = 2
        self.population = []
        self.new_population = []


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


    def genetic_algorithm(self):
        self.construct_population()

        for i in range(0, len(self.solution.population), 2):
            self.crossover(self.solution.population[i], self.solution.population[i + 1])


    def construct_population(self):
        for i in range(self.solution.size_population):
            self.greedy_construction(init_randomly=True)

            self.solution.population.append(self.solution.bin_packs)


    def greedy_construction(self, init_randomly=False, index_to_start_reconstruction=None):
        # If this index is not None, we need to destroy the solution starting at this index
        if index_to_start_reconstruction != None:
            del self.solution.bin_packs[index_to_start_reconstruction:]
        else:
            self.solution.bin_packs = []

        data_indexes = list(range(len(self.instance.data_values)))

        if init_randomly:
            np.random.shuffle(data_indexes)

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

        # Elements of parents to insert in opposite child
        bins_split_first_child = np.array(copy_first_parent[first_parent_splits[0]:first_parent_splits[1]])
        bins_split_second_child = np.array(copy_second_parent[second_parent_splits[0]:second_parent_splits[1]])

        # Create first child
        self.solution.bin_packs = np.copy(copy_first_parent)
        self._delete_bins_with_duplicated_elements(bins_split_second_child)
        first_child = np.concatenate([self.solution.bin_packs[:first_parent_splits[0]],
                                      bins_split_second_child,
                                      self.solution.bin_packs[first_parent_splits[0]:]])
        self.solution.new_population.append(first_child)

        # Create second child
        self.solution.bin_packs = np.copy(copy_second_parent)
        self._delete_bins_with_duplicated_elements(bins_split_first_child)
        second_child = np.concatenate([self.solution.bin_packs[:second_parent_splits[0]],
                                       bins_split_first_child,
                                       self.solution.bin_packs[second_parent_splits[0]:]])
        self.solution.new_population.append(second_child)


    def _delete_bins_with_duplicated_elements(self, new_gene_bins):
        # Find bins with repeated elements
        bin_with_equal_elements = []
        for new_gene_bin in new_gene_bins:
            for new_gene_element in new_gene_bin.samples:
                equal_element = False
                for current_bin in self.solution.bin_packs:
                    for current_element in current_bin.samples:
                        if current_element == new_gene_element:
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


    def local_search(self):
        note_used_samples = []
        for bin in self.solution.bin_packs:
            for i in range(len(bin.samples) - 2):
                changed_bin = False
                one_element = self.instance.data_values[bin.samples[i]]
                two_elements = one_element + self.instance.data_values[bin.samples[i + 1]]
                three_elements = two_elements + self.instance.data_values[bin.samples[i + 2]]
                for j in range(len(self.solution.eliminated_elements) - 1):
                    index_first = self.solution.eliminated_elements[j]
                    index_second = self.solution.eliminated_elements[j + 1]
                    one_excluded_element = self.instance.data_values[index_first]
                    two_excluded_element = one_excluded_element + self.instance.data_values[index_second]
                    if one_excluded_element >= three_elements:
                        partial_used_space = bin.used_space - three_elements + one_excluded_element
                        if self.instance.bin_constraints[0] >= partial_used_space[0][0] and self.instance.bin_constraints[1] >= partial_used_space[0][1]:
                            changed_bin = True
                            self.solution.bin_packs[bin].add_sample([index_first])
                            note_used_samples.append(bin.samples[i])
                            note_used_samples.append(bin.samples[i + 1])
                            note_used_samples.append(bin.samples[i + 2])
                            self.solution.bin_packs[bin].remove_sample([bin.samples[i], bin.samples[i + 1], bin.samples[i + 2]])
                            break
                    elif one_excluded_element >= two_elements:
                        partial_used_space = bin.used_space - two_elements + one_excluded_element
                        if self.instance.bin_constraints[0] >= partial_used_space[0][0] and self.instance.bin_constraints[1] >= partial_used_space[0][1]:
                            changed_bin = True
                            self.solution.bin_packs[bin].add_sample([index_first])
                            note_used_samples.append(bin.samples[i])
                            note_used_samples.append(bin.samples[i + 1])
                            self.solution.bin_packs[bin].remove_sample([bin.samples[i], bin.samples[i + 1]])
                            break
                    elif two_excluded_element >= three_elements:
                        partial_used_space = bin.used_space - three_elements + two_excluded_element
                        if self.instance.bin_constraints[0] >= partial_used_space[0][0] and self.instance.bin_constraints[1] >= partial_used_space[0][1]:
                            changed_bin = True
                            self.solution.bin_packs[bin].add_sample([index_first, index_second])
                            note_used_samples.append(bin.samples[i])
                            note_used_samples.append(bin.samples[i + 1])
                            note_used_samples.append(bin.samples[i + 2])
                            self.solution.bin_packs[bin].remove_sample([bin.samples[i], bin.samples[i + 1], bin.samples[i + 2]])
                        break
                    elif two_excluded_element >= two_elements:
                        partial_used_space = bin.used_space - two_elements + two_excluded_element
                        if self.instance.bin_constraints[0] >= partial_used_space[0][0] and self.instance.bin_constraints[1] >= partial_used_space[0][1]:
                            changed_bin = True
                            self.solution.bin_packs[bin].add_sample([index_first, index_second])
                            note_used_samples.append(bin.samples[i])
                            note_used_samples.append(bin.samples[i + 1])
                            self.solution.bin_packs[bin].remove_sample([bin.samples[i], bin.samples[i + 1]])
                        break
                    elif one_excluded_element >= one_element:
                        partial_used_space = bin.used_space - one_element + one_excluded_element
                        if self.instance.bin_constraints[0] >= partial_used_space[0][0] and self.instance.bin_constraints[1] >= partial_used_space[0][1]:
                            changed_bin = True
                            self.solution.bin_packs[bin].add_sample([index_first])
                            note_used_samples.append(bin.samples[i])
                            self.solution.bin_packs[bin].remove_sample([bin.samples[i]])
                            break

                if changed_bin == True:
                    break

        note_used_samples = note_used_samples + self.solution.eliminated_elements[:]
        self.solution.eliminated_elements = []
        self.first_fit(note_used_samples)


    def first_fit(self, sample_ids):
        pass
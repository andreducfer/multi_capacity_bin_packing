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
        self.bin_packs = np.array([], dtype=object)
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

        print('Hello World!')


    def construct_population(self):
        for i in range(self.solution.size_population):
            self.greedy_construction(init_randomly=True)

            self.solution.population.append(np.copy(self.solution.bin_packs))

            self.solution.bin_packs = np.array([], dtype=object)


    def greedy_construction(self, init_randomly=False, data_indexes=None):

        if data_indexes is None:
            data_indexes = list(range(len(self.instance.data_values)))

        if init_randomly:
            np.random.shuffle(data_indexes)

        while len(data_indexes) > 0:
            new_bin = self._create_and_fill_bin(data_indexes)

            self.solution.bin_packs = np.append(self.solution.bin_packs, new_bin)

            data_indexes_to_new_bin = new_bin.samples

            for data_index_to_new_bin in data_indexes_to_new_bin:
                data_indexes.remove(data_index_to_new_bin)


    def _create_and_fill_bin(self, data_indexes):
        new_bin = Bin(self.instance)
        for data_index in data_indexes:
            if new_bin.verify_capacity(data_index):
                new_bin.add_sample([data_index])

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
        self.local_search()
        first_child = np.concatenate([self.solution.bin_packs[:first_parent_splits[0]],
                                      bins_split_second_child,
                                      self.solution.bin_packs[first_parent_splits[0]:]])
        self.solution.new_population.append(first_child)

        # Create second child
        self.solution.bin_packs = np.copy(copy_second_parent)
        self._delete_bins_with_duplicated_elements(bins_split_first_child)
        self.local_search()
        second_child = np.concatenate([self.solution.bin_packs[:second_parent_splits[0]],
                                       bins_split_first_child,
                                       self.solution.bin_packs[second_parent_splits[0]:]])
        self.solution.new_population.append(second_child)


    def _delete_bins_with_duplicated_elements(self, new_gene_bins):
        # Find bins with repeated elements
        bins_to_delete = []
        for new_gene_bin in new_gene_bins:
            if len(bins_to_delete) > 0:
                for bin_to_delete in bins_to_delete:
                    self.solution.bin_packs = np.delete(self.solution.bin_packs, bin_to_delete)
                bins_to_delete = []
            for new_gene_element in new_gene_bin.samples:
                equal_element = False
                for index_current_bin, current_bin in enumerate(self.solution.bin_packs):
                    for current_element in current_bin.samples:
                        if current_element == new_gene_element:
                            equal_element = True
                            break
                    if equal_element == True:
                        bins_to_delete.append(index_current_bin)
                        for element_to_eliminate in current_bin.samples:
                            self.solution.eliminated_elements.append(element_to_eliminate)
                        break
                if equal_element == True:
                    break


    def local_search(self):
        # TODO adicionar evaluate quando meche na BIN
        not_used_samples = []
        for index, bin in enumerate(self.solution.bin_packs):
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
                    if two_excluded_element[0] >= three_elements[0] and two_excluded_element[1] >= three_elements[1]:
                        partial_used_space = bin.used_space - three_elements + two_excluded_element
                        if self.instance.bin_constraints[0] >= partial_used_space[0][0] and self.instance.bin_constraints[1] >= partial_used_space[0][1]:
                            changed_bin = True
                            self.solution.bin_packs[index].add_sample([index_first, index_second])
                            not_used_samples.append(bin.samples[i])
                            not_used_samples.append(bin.samples[i + 1])
                            not_used_samples.append(bin.samples[i + 2])
                            self.solution.bin_packs[index].remove_sample([bin.samples[i], bin.samples[i + 1], bin.samples[i + 2]])
                            self.solution.eliminated_elements.remove(index_first)
                            self.solution.eliminated_elements.remove(index_second)
                            break
                    elif one_excluded_element[0] >= three_elements[0] and one_excluded_element[1] >= three_elements[1]:
                        partial_used_space = bin.used_space - three_elements + one_excluded_element
                        if self.instance.bin_constraints[0] >= partial_used_space[0][0] and self.instance.bin_constraints[1] >= partial_used_space[0][1]:
                            changed_bin = True
                            self.solution.bin_packs[index].add_sample([index_first])
                            not_used_samples.append(bin.samples[i])
                            not_used_samples.append(bin.samples[i + 1])
                            not_used_samples.append(bin.samples[i + 2])
                            self.solution.bin_packs[index].remove_sample([bin.samples[i], bin.samples[i + 1], bin.samples[i + 2]])
                            self.solution.eliminated_elements.remove(index_first)
                            break
                    elif two_excluded_element[0] >= two_elements[0] and two_excluded_element[1] >= two_elements[1]:
                        partial_used_space = bin.used_space - two_elements + two_excluded_element
                        if self.instance.bin_constraints[0] >= partial_used_space[0][0] and self.instance.bin_constraints[1] >= partial_used_space[0][1]:
                            changed_bin = True
                            self.solution.bin_packs[index].add_sample([index_first, index_second])
                            not_used_samples.append(bin.samples[i])
                            not_used_samples.append(bin.samples[i + 1])
                            self.solution.bin_packs[index].remove_sample([bin.samples[i], bin.samples[i + 1]])
                            self.solution.eliminated_elements.remove(index_first)
                            self.solution.eliminated_elements.remove(index_second)
                            break
                    elif one_excluded_element[0] >= two_elements[0] and one_excluded_element[1] >= two_elements[1]:
                        partial_used_space = bin.used_space - two_elements + one_excluded_element
                        if self.instance.bin_constraints[0] >= partial_used_space[0][0] and self.instance.bin_constraints[1] >= partial_used_space[0][1]:
                            changed_bin = True
                            self.solution.bin_packs[index].add_sample([index_first])
                            not_used_samples.append(bin.samples[i])
                            not_used_samples.append(bin.samples[i + 1])
                            self.solution.bin_packs[index].remove_sample([bin.samples[i], bin.samples[i + 1]])
                            self.solution.eliminated_elements.remove(index_first)
                            break
                    elif one_excluded_element[0] >= one_element[0] and one_excluded_element[1] >= one_element[1]:
                        partial_used_space = bin.used_space - one_element + one_excluded_element
                        if self.instance.bin_constraints[0] >= partial_used_space[0][0] and self.instance.bin_constraints[1] >= partial_used_space[0][1]:
                            changed_bin = True
                            self.solution.bin_packs[index].add_sample([index_first])
                            not_used_samples.append(bin.samples[i])
                            self.solution.bin_packs[index].remove_sample([bin.samples[i]])
                            self.solution.eliminated_elements.remove(index_first)
                            break

                if changed_bin == True:
                    break

        not_used_samples = not_used_samples + self.solution.eliminated_elements[:]
        self.solution.eliminated_elements = []
        not_used_samples = sorted(not_used_samples)
        self.greedy_construction(data_indexes=not_used_samples)

from data_handler import Instance

import numpy as np
from copy import deepcopy

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
        self.eliminated_elements = []
        self.size_population = 10
        self.size_crossover_to_survive = 5
        self.size_local_search_to_survive = 3
        self.size_population_to_survive = 2

        self.best_fitness = 0
        self.fitness_population = np.array([], dtype=float)
        self.fitness_new_population = np.array([], dtype=float)
        self.fitness_local_search_population = np.array([], dtype=float)

        self.best_bin_pack = [] #np.array([], dtype=object)
        self.bin_packs = [] #np.array([], dtype=object)
        self.population = [] #np.array([], dtype=object)
        self.new_population = [] #np.array([], dtype=object)
        self.local_search_population = [] #np.array([], dtype=object)


    def get_num_bins(self):
        return len(self.bin_packs)


    def calculate_fitness_person(self, new_population=False, local_search=False, fitness_index_to_replace=None):
        sum_bins_fitness = 0
        for bin in self.bin_packs:
            sum_bins_fitness = sum_bins_fitness + bin.fitness

        fitness_person = sum_bins_fitness / len(self.bin_packs)

        if new_population:
            self.fitness_new_population = np.append(self.fitness_new_population, fitness_person)
        elif local_search and fitness_index_to_replace is not None:
            self.fitness_local_search_population[fitness_index_to_replace] = fitness_person
        elif local_search:
            #assert fitness_index_to_replace is not None
            self.fitness_local_search_population = np.append(self.fitness_local_search_population, fitness_person)
        else:
            self.fitness_population = np.append(self.fitness_population, fitness_person)


    def sort_population(self):
        if len(self.population) > 1:
            index_sorted_by_fitness = np.argsort(self.fitness_population)[::-1]

            self.population = [self.population[i] for i in index_sorted_by_fitness]

            self.fitness_population = self.fitness_population[index_sorted_by_fitness]

        if len(self.new_population) > 1:
            index_sorted_by_fitness = np.argsort(self.fitness_new_population)[::-1]

            self.new_population = [self.new_population[i] for i in index_sorted_by_fitness]

            self.fitness_new_population = self.fitness_new_population[index_sorted_by_fitness]

        if len(self.local_search_population) > 1:
            index_sorted_by_fitness = np.argsort(self.fitness_local_search_population)[::-1]

            self.local_search_population = [self.local_search_population[i] for i in index_sorted_by_fitness]

            self.fitness_local_search_population = self.fitness_local_search_population[index_sorted_by_fitness]


    def print_and_export(self):
        pack = []

        for i, bin in enumerate(self.best_bin_packs):
            bin_info = 'Bin ' + str(i) + 'samples: '
            for sample in bin.samples:
                bin_template = str(sample) + ', '


class Genetic:
    def __init__(self, instance, solution):
        if not isinstance(instance, Instance):
            raise TypeError("Error: instance variable is not data_handler.Instance. Type: " + type(instance))
        if not isinstance(solution, Solution):
            raise TypeError("ERROR: solution variable is not decision_tree.Solution. Type: " + type(solution))
        self.instance = instance
        self.solution = solution


    def genetic_algorithm(self):
        self.construct_population()

        for k in range(10):

            for i in range(0, len(self.solution.population), 2):
                self.crossover(self.solution.population[i], self.solution.population[i + 1])

            self.construction_of_local_search_population()
            self.construction_of_new_population()

            self.find_best_solution()

        print("Best fitness: %.2f" % self.solution.best_fitness)


    def find_best_solution(self):
        best_current_solution_index = np.argmax(self.solution.fitness_population)
        best_current_solution_fitness = self.solution.fitness_population[best_current_solution_index]

        if(best_current_solution_fitness >= self.solution.best_fitness):
            self.solution.best_bin_pack = deepcopy(self.solution.population[best_current_solution_index])
            self.solution.best_fitness = best_current_solution_fitness


    def construct_population(self):
        for i in range(self.solution.size_population):
            if i == 0:
                init_randomly = False
            else:
                init_randomly = True

            self.first_fit(init_randomly=init_randomly)
            print('Number bins: ' + str(self.solution.get_num_bins()))

            self.solution.population.append(deepcopy(self.solution.bin_packs))

            self.solution.bin_packs = []

        self.solution.sort_population()


    def first_fit(self, init_randomly=False, data_indexes=None, new_population=False, local_search=False, fitness_index_to_replace=None):

        if data_indexes is None:
            data_indexes = list(range(len(self.instance.data_values)))

        if init_randomly:
            np.random.shuffle(data_indexes)

        while len(data_indexes) > 0:
            # Create and fill a bin
            new_bin = Bin(self.instance)
            for data_index in data_indexes:
                if new_bin.verify_capacity(data_index):
                    new_bin.add_sample([data_index])
            new_bin.evaluate()

            self.solution.bin_packs.append(new_bin)

            data_indexes_to_new_bin = new_bin.samples

            for data_index_to_new_bin in data_indexes_to_new_bin:
                data_indexes.remove(data_index_to_new_bin)

        self.solution.calculate_fitness_person(new_population=new_population, local_search=local_search, fitness_index_to_replace=fitness_index_to_replace)


    def crossover(self, first_parent, second_parent):
        copy_first_parent = deepcopy(first_parent)
        copy_second_parent = deepcopy(second_parent)

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
        bins_split_first_child = deepcopy(copy_first_parent[first_parent_splits[0]:first_parent_splits[1]])
        bins_split_second_child = deepcopy(copy_second_parent[second_parent_splits[0]:second_parent_splits[1]])

        # Create first child
        self.solution.bin_packs = deepcopy(copy_first_parent)
        self._delete_bins_with_duplicated_elements(bins_split_second_child)
        self.local_search(new_population=True)

        #first_child = np.concatenate([self.solution.bin_packs[:first_parent_splits[0]],
        #                              bins_split_second_child,
        #                              self.solution.bin_packs[first_parent_splits[0]:]])
        first_child = self.solution.bin_packs[:first_parent_splits[0]] + bins_split_second_child + self.solution.bin_packs[first_parent_splits[0]:]

        #self.solution.new_population = np.append(self.solution.new_population, first_child)
        self.solution.new_population.append(first_child)

        # Create second child
        self.solution.bin_packs = deepcopy(copy_second_parent)
        self._delete_bins_with_duplicated_elements(bins_split_first_child)
        self.local_search(new_population=True)

        #second_child = np.concatenate([self.solution.bin_packs[:second_parent_splits[0]],
        #                               bins_split_first_child,
        #                               self.solution.bin_packs[second_parent_splits[0]:]])
        second_child = self.solution.bin_packs[:second_parent_splits[0]] + bins_split_first_child + self.solution.bin_packs[second_parent_splits[0]:]

        #self.solution.new_population = np.append(self.solution.new_population, second_child)
        self.solution.new_population.append(second_child)


    def _delete_bins_with_duplicated_elements(self, new_gene_bins):
        # Find bins with repeated elements
        bins_to_delete = []
        for new_gene_bin in new_gene_bins:
            if len(bins_to_delete) > 0:
                for bin_to_delete in bins_to_delete:
                    del self.solution.bin_packs[bin_to_delete]
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


    def local_search(self, new_population=False, local_search=False, fitness_index_to_replace=None):
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
                            self.solution.bin_packs[index].evaluate()
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
                            self.solution.bin_packs[index].evaluate()
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
                            self.solution.bin_packs[index].evaluate()
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
                            self.solution.bin_packs[index].evaluate()
                            break
                    elif one_excluded_element[0] >= one_element[0] and one_excluded_element[1] >= one_element[1]:
                        partial_used_space = bin.used_space - one_element + one_excluded_element
                        if self.instance.bin_constraints[0] >= partial_used_space[0][0] and self.instance.bin_constraints[1] >= partial_used_space[0][1]:
                            changed_bin = True
                            self.solution.bin_packs[index].add_sample([index_first])
                            not_used_samples.append(bin.samples[i])
                            self.solution.bin_packs[index].remove_sample([bin.samples[i]])
                            self.solution.eliminated_elements.remove(index_first)
                            self.solution.bin_packs[index].evaluate()
                            break

                if changed_bin == True:
                    break

        not_used_samples = not_used_samples + self.solution.eliminated_elements[:]
        self.solution.eliminated_elements = []
        not_used_samples = sorted(not_used_samples)
        self.first_fit(data_indexes=not_used_samples, new_population=new_population, local_search=local_search, fitness_index_to_replace=fitness_index_to_replace)


    def construction_of_new_population(self):
        self.solution.sort_population()

        crossover_population_to_survive = deepcopy(self.solution.new_population[:self.solution.size_crossover_to_survive])
        fitness_crossover_to_survive = np.copy(self.solution.fitness_new_population[:self.solution.size_crossover_to_survive])

        local_search_population_to_survive = deepcopy(self.solution.local_search_population[:self.solution.size_local_search_to_survive])
        fitness_local_search_population_to_survive = np.copy(self.solution.fitness_local_search_population[:self.solution.size_local_search_to_survive])

        old_population_to_survive = deepcopy(self.solution.population[:self.solution.size_population_to_survive])
        fitness_old_population_to_survive = np.copy(self.solution.fitness_population[:self.solution.size_population_to_survive])

        # self.solution.population = np.concatenate([crossover_population_to_survive, local_search_population_to_survive, old_population_to_survive])
        self.solution.population = crossover_population_to_survive + local_search_population_to_survive + old_population_to_survive

        self.solution.fitness_population = np.array([], dtype=float)
        self.solution.fitness_population = np.concatenate([fitness_crossover_to_survive, fitness_local_search_population_to_survive, fitness_old_population_to_survive])


    def construction_of_local_search_population(self):
        random_indexes = np.random.randint(self.solution.size_population, size=self.solution.size_local_search_to_survive)

        self.solution.local_search_population = []
        self.solution.fitness_local_search_population = np.array([], dtype=float)

        new_local_search_population = []
        new_local_search_population_fitness = np.array([], dtype=float)

        for index in random_indexes:
            self.solution.local_search_population.append(deepcopy(self.solution.population[index]))
            self.solution.fitness_local_search_population = np.append(self.solution.fitness_local_search_population, self.solution.fitness_population[index])


        for i, bin_pack in enumerate(self.solution.local_search_population):
            best_binpack = deepcopy(bin_pack)
            best_binpack_fitness = np.copy(self.solution.fitness_local_search_population[i])
            random_indexes_destroy_solution = np.random.randint(len(bin_pack), size=self.solution.size_local_search_to_survive)

            for index_destroy_ahead in random_indexes_destroy_solution:
                #self.solution.bin_packs = []
                self.solution.bin_packs = deepcopy(bin_pack)
                self.solution.eliminated_elements = []
                for index_bin in range(index_destroy_ahead, len(self.solution.bin_packs)):
                    current_bin = self.solution.bin_packs[index_bin]
                    self.solution.eliminated_elements.extend(current_bin.samples)

                self.solution.bin_packs = self.solution.bin_packs[:index_destroy_ahead]

                self.local_search(local_search=True, new_population=False, fitness_index_to_replace=i)

                if best_binpack_fitness <= self.solution.fitness_local_search_population[i]:
                    best_binpack = deepcopy(self.solution.bin_packs)
                    best_binpack_fitness = np.copy(self.solution.fitness_local_search_population[i])

            # new_local_search_population.append(best_binpack)
            # new_local_search_population_fitness = np.append(new_local_search_population_fitness, best_binpack_fitness)

            self.solution.local_search_population[i] = deepcopy(best_binpack)
            self.solution.fitness_local_search_population[i] = np.copy(best_binpack_fitness)
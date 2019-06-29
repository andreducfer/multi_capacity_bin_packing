from data_handler import Instance
from bin_packing import Solution
from bin_packing import Genetic

import argparse
from os import listdir
from os.path import join
import time
from datetime import datetime
import random

default_seed_list = [0, 1, 2, 3, 4]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi capacity bin packing problem")

    parser.add_argument('-o', '--solution_dir', default='solution', help="Directory to save solutions.")
    parser.add_argument('-d', '--dataset_dir', default='dataset', help='Directory containing dataset files.')
    parser.add_argument('-f', '--dataset_file', help="Dataset file to be processed.")
    parser.add_argument('-s', '--seed', type=int, nargs='+', default=default_seed_list, help='Seed for the random function.')
    parser.add_argument('-m', '--max_execution_time_in_seconds', default=300, type=int, help="Maximum execution time for each instance. In seconds.")

    args = parser.parse_args()

    if args.dataset_file is not None:
        dataset_files = [args.dataset_file]
    else:
        dataset_files = sorted(listdir(args.dataset_dir))

    for dataset_file in dataset_files:
        print("Dataset: %s" % dataset_file)
        for seed in args.seed:
            # Limit of time to run the algorithm
            time_limit = time.time() + args.max_execution_time_in_seconds

            # Set a specific seed
            random.seed(seed)

            # Path where is the dataset
            dataset_path = join(args.dataset_dir, dataset_file)
            dataset_name = dataset_file.rstrip('.txt')

            # Path and model of solution file
            time_now = datetime.now().strftime("%Y%m%d%H%M%S")
            solution_file = "%s-%s-seed_%d.txt" % (dataset_name, time_now, seed)
            solution_path = join(args.solution_dir, solution_file)

            # Initialization of instance
            instance = Instance(dataset_path, solution_path, dataset_name, seed, args.max_execution_time_in_seconds)

            solution = Solution(instance)

            greedy = Genetic(instance, solution)
            greedy.genetic_algorithm()

            print("'%d' bins for seed '%d'" % (greedy.solution.get_num_bins(), seed))

        print("##########################################")
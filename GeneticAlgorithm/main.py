import time
import itertools
from csv_utils import export_results_to_csv
from data import prepare_data_easy, prepare_data_hard, prepare_data_hard2, prepare_data_medium
from genetic_algorithm import genetic_algorithm, grid_search_ga
from genetic_utils import calculate_fitness, generate_random_chromosome
from models import Chromosome
from utils import display_schedule, display_schedule_table, print_best_config, show_schedule_gui


if __name__ == "__main__":
    output_csv_file: str = "genetic_algorithm_grid_search_results.csv"

    groups_list, lessons_list, rooms_list, slots_list = prepare_data_medium()
    initial_chromosome: Chromosome = generate_random_chromosome(groups_list, lessons_list, rooms_list, slots_list)

    fitness_parameters_grid = [
        {
            "init_score": 1000,
            "group_conflict_penalty": group_penalty,
            "teacher_conflict_penalty": teacher_penalty,
            "room_conflict_penalty": room_penalty,
            "gap_penalty": gap_penalty
        }
        for group_penalty, teacher_penalty, room_penalty, gap_penalty in itertools.product(
            [100, 200],
            [50, 150],
            [50, 100],
            [0, 5]
        )
    ]

    ga_parameters_grid = [
        {
            "population_size": population_size,
            "generations": num_generations,
            "crossover_rate": crossover_prob,
            "mutation_rate": mutation_prob,
            "patience": patience_value
        }
        for population_size, num_generations, crossover_prob, mutation_prob, patience_value in itertools.product(
            [50, 75],
            [500, 1000],
            [0.8, 0.9],
            [0.05, 0.1],
            [50, 100, 200] 
        )
    ]

    best_chromosome, best_fitness_score, best_configuration, all_results = grid_search_ga(
        groups_list, lessons_list, rooms_list, slots_list,
        fitness_parameters_grid, ga_parameters_grid
    )

    print_best_config(best_fitness_score, best_configuration)

    export_results_to_csv(all_results, output_csv_file)

    display_schedule(best_chromosome)
    show_schedule_gui(best_chromosome)

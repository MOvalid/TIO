import itertools
from csv_utils import export_results_to_csv
from data import prepare_data_easy, prepare_data_hard, prepare_data_hard2, prepare_data_medium
from genetic_algorithm import genetic_algorithm, grid_search_ga, repeat_best_config
from genetic_utils import calculate_fitness, generate_random_chromosome
from models import Chromosome
from utils import analyze_and_plot_best_repeat, display_schedule, display_schedule_table, plot_fitness_progression, print_best_config, show_schedule_gui, summarize_ga_results
from math import inf


if __name__ == "__main__":
    output_csv_file: str = "genetic_algorithm_grid_search_results2.csv"

    groups_list, lessons_list, rooms_list, slots_list = prepare_data_medium()
    initial_chromosome: Chromosome = generate_random_chromosome(groups_list, lessons_list, rooms_list, slots_list)

    # fitness_parameters_grid = [
    #     {
    #         "init_score": 100,
    #         "group_conflict_penalty": group_penalty,
    #         "teacher_conflict_penalty": teacher_penalty,
    #         "room_conflict_penalty": room_penalty,
    #         "gap_penalty_short": gap_penalty_short,
    #         "gap_penalty_long": gap_penalty_long
    #     }
    #     for group_penalty, teacher_penalty, room_penalty, gap_penalty_short, gap_penalty_long in itertools.product(
    #         [100, 200],
    #         [50, 150],
    #         [50, 100],
    #         [10, 20],
    #         [20, 50]
    #     )
    # ]

    # group_conflict_penalty: int = 100,
    # teacher_conflict_penalty: int = 100,
    # room_conflict_penalty: int = 100,
    # group_gap_penalty_short: int = 20,
    # group_gap_penalty_long: int = 40,
    # teacher_gap_penalty_short: int = 5,
    # teacher_gap_penalty_long: int = 10

    fitness_parameters_grid = [
        {
            "init_score": 500,
            "group_conflict_penalty": group_penalty,
            "teacher_conflict_penalty": teacher_penalty,
            "room_conflict_penalty": room_penalty,
            "group_gap_penalty_short": group_gap_penalty_short,
            "group_gap_penalty_long": group_gap_penalty_long,
            "teacher_gap_penalty_short": teacher_gap_penalty_short,
            "teacher_gap_penalty_long": teacher_gap_penalty_long
        }
        for group_penalty, teacher_penalty, room_penalty, group_gap_penalty_short, group_gap_penalty_long, teacher_gap_penalty_short, teacher_gap_penalty_long in itertools.product(
            [inf],
            [inf],
            [inf],
            [10, 20],
            [20, 50],
            [10, 20],
            [20, 50],
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
        fitness_parameters_grid, ga_parameters_grid, False
    )

    print_best_config(best_fitness_score, best_configuration)
    export_results_to_csv(all_results, output_csv_file)
    display_schedule(best_chromosome)
    show_schedule_gui(best_chromosome)
    summarize_ga_results(output_csv_file)


    # --- DODATKOWE POWTÓRZENIA NAJLEPSZEJ KONFIGURACJI ---

    fitness_matrix = repeat_best_config(
        repeats=20,
        groups_list=groups_list,
        lessons_list=lessons_list,
        rooms_list=rooms_list,
        slots_list=slots_list,
        best_configuration=best_configuration,
        genetic_algorithm_func=genetic_algorithm
    )

    analyze_and_plot_best_repeat(
        fitness_matrix=fitness_matrix,
        plot_func=plot_fitness_progression
    )

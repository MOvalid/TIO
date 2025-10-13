import itertools
from csv_utils import save_results_to_csv
from data import prepare_data_medium
from genetic_algorithm import genetic_algorithm
from genetic_utils import generate_random_chromosome
from utils import plot_fitness_split_by_crossover_and_pop_gen
from math import inf


if __name__ == "__main__":
    output_csv_file = "genetic_algorithm_grid_search_results2.csv"

    groups_list, lessons_list, rooms_list, slots_list = prepare_data_medium()
    initial_chromosome = generate_random_chromosome(groups_list, lessons_list, rooms_list, slots_list)

    default_fitness_params = {
        "init_score": 1000,
        "group_conflict_penalty": inf,
        "teacher_conflict_penalty": inf,
        "room_conflict_penalty": inf,
        "group_gap_penalty_short": 20,
        "group_gap_penalty_long": 40,
        "teacher_gap_penalty_short": 5,
        "teacher_gap_penalty_long": 10,
        "concentration_bonus": 50
    }

    ga_parameters_grid = [
        {
            "population_size": population_size,
            "generations": num_generations,
            "crossover_rate": crossover_prob,
            "mutation_rate": mutation_prob,
            "patience": patience_value
        }
        # for population_size, num_generations, crossover_prob, mutation_prob, patience_value in itertools.product(
        #     [75, 100],
        #     [500, 1000],
        #     [0.6, 0.7, 0.8, 0.9],
        #     [0.05, 0.1, 0.15],
        #     [None]
        # )
        for population_size, num_generations, crossover_prob, mutation_prob, patience_value in itertools.product(
            [100, 125],
            [1000],
            [0.8, 0.9],
            [0.1],
            [None]
        )
    ]

    repeats = 1
    fitness_histories = {}
    results = []

    for ga_params in ga_parameters_grid:
        best_run_history = None
        best_run_max_fitness = -float('inf')

        print(f"Testing GA params: {ga_params}")

        for run in range(repeats):
            print(f" Run {run+1}/{repeats}")
            _, _, fitness_history = genetic_algorithm(
                groups=groups_list,
                lessons=lessons_list,
                rooms=rooms_list,
                slots=slots_list,
                fitness_kwargs=default_fitness_params,
                record_all_generations=True,
                **ga_params
            )
            max_fitness = max(fitness_history) if fitness_history else -float('inf')
            if max_fitness > best_run_max_fitness:
                best_run_max_fitness = max_fitness
                best_run_history = fitness_history

        fitness_histories[tuple(sorted(ga_params.items()))] = best_run_history
        results.append({
        **ga_params,
        "max_fitness": best_run_max_fitness,
        "fitness_history": best_run_history
    })

    plot_fitness_split_by_crossover_and_pop_gen(fitness_histories)
    save_results_to_csv(results)

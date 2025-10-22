import itertools
from csv_utils import save_results_to_csv
from data import prepare_data_hard, prepare_data_medium
from genetic_algorithm import (
    genetic_algorithm,
    tournament_selection,
    roulette_wheel_selection,
    rank_selection
)
from genetic_utils import generate_random_chromosome
from utils import plot_fitness_split_by_crossover_and_pop_gen
from math import inf


if __name__ == "__main__":
    output_csv_file = "genetic_algorithm_selection_comparison.csv"

    groups_list, lessons_list, rooms_list, slots_list = prepare_data_medium()
    initial_chromosome = generate_random_chromosome(groups_list, lessons_list, rooms_list, slots_list)

    default_fitness_params = {
        "init_score": 0,
        "group_conflict_penalty": inf,
        "teacher_conflict_penalty": inf,
        "room_conflict_penalty": inf,
        "group_gap_penalty_short": 20,
        "group_gap_penalty_long": 40,
        "teacher_gap_penalty_short": 5,
        "teacher_gap_penalty_long": 10,
        "concentration_bonus": 50
    }

    selection_methods = {
        "tournament": tournament_selection,
        "roulette": roulette_wheel_selection,
        "rank": rank_selection
    }

    ga_parameters_grid = [
        {
            "population_size": population_size,
            "generations": num_generations,
            "crossover_rate": crossover_prob,
            "mutation_rate": mutation_prob,
            "patience": patience_value,
            "selection_function": selection_method,
            "elite_size": elite_size
        }
        for population_size, num_generations, crossover_prob, mutation_prob, patience_value, selection_method, elite_size in itertools.product(
            [100],  # population size
            [500],  # generations
            [0.8],  # crossover rate
            [0.1],  # mutation rate
            [None],  # patience
            selection_methods.values(),  # selection methods
            [0, 2, 5]  # elite size
        )
    ]

    repeats = 3
    fitness_histories = {}
    results = []

    for ga_params in ga_parameters_grid:
        selection_name = [k for k, v in selection_methods.items() 
                         if v == ga_params["selection_function"]][0]
        print(f"\nTesting with selection method: {selection_name}")
        print(f"Elite size: {ga_params['elite_size']}")
        
        best_run_history = None
        best_run_max_fitness = -float('inf')

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

        # Store results with selection method name instead of function
        params_for_storage = ga_params.copy()
        params_for_storage["selection_method"] = selection_name
        del params_for_storage["selection_function"]
        
        key = tuple(sorted((k, str(v)) for k, v in params_for_storage.items()))
        fitness_histories[key] = best_run_history
        
        results.append({
            **params_for_storage,
            "max_fitness": best_run_max_fitness,
            "fitness_history": best_run_history
        })

    plot_fitness_split_by_crossover_and_pop_gen(fitness_histories)
    save_results_to_csv(results, output_csv_file)

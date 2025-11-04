import os
import itertools
import time
import pandas as pd
import numpy as np
from datetime import datetime
from csv_utils import save_results_to_csv
from data import prepare_data_medium
from genetic_algorithm import (
    genetic_algorithm,
    tournament_selection,
    roulette_wheel_selection,
    rank_selection,
    elitist_selection
)
from genetic_utils import generate_random_chromosome
from genetic_operators import (
    one_point_crossover,
    two_point_crossover,
    uniform_crossover,
    mutate,
    swap_mutation,
    group_reverse_mutation
)
import matplotlib.pyplot as plt
from math import inf


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_comparison_with_std(histories_list, labels, title, filename_prefix):
    plt.figure(figsize=(12, 8))
    
    for label, histories in zip(labels, histories_list):
        histories_array = np.array(histories)
        histories_array = np.nan_to_num(histories_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        mean_fitness = np.nanmean(histories_array, axis=0)
        std_fitness = np.nanstd(histories_array, axis=0)
        
        mean_fitness = np.nan_to_num(mean_fitness, nan=0.0, posinf=0.0, neginf=0.0)
        std_fitness = np.nan_to_num(std_fitness, nan=0.0, posinf=0.0, neginf=0.0)
        
        plt.plot(mean_fitness, label=label)

        plt.fill_between(
            range(len(mean_fitness)),
            mean_fitness - std_fitness,
            mean_fitness + std_fitness,
            alpha=0.2
        )
    
    plt.title(title)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    
    ensure_directory_exists('./diagrams')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./diagrams/{filename_prefix}_{timestamp}.png"
    plt.savefig(filename)
    plt.close()

def create_comparison_table(results_dict, runs):
    stats = []
    for variant, data in results_dict.items():
        final_fitnesses = [history[-1] for history in data]
        stats.append({
            'Variant': variant,
            'Mean Fitness': np.mean(final_fitnesses),
            'Std Dev': np.std(final_fitnesses),
            'Max Fitness': np.max(final_fitnesses),
            'Min Fitness': np.min(final_fitnesses),
            'Runs': runs
        })
    return pd.DataFrame(stats)

if __name__ == "__main__":
    groups_list, lessons_list, rooms_list, slots_list = prepare_data_medium()
    RUNS = 5

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

    # Test 1: Compare crossover operators with different rates
    crossover_rates = [0.7, 0.8, 0.9, 0.95]
    mutation_rates = [0.05, 0.1, 0.15]
    crossover_comparison = {
        "one_point": one_point_crossover,
        "two_point": two_point_crossover,
        "uniform": lambda p1, p2: uniform_crossover(p1, p2, check_validity=True)
    }

    crossover_histories = {}
    for crossover_name, crossover_func in crossover_comparison.items():
        for mutation_rate in mutation_rates:
            for crossover_rate in crossover_rates:
                variant_name = f"{crossover_name}_{mutation_rate}_crossover_{crossover_rate}"
                crossover_histories[variant_name] = []
                print(f"\nTesting crossover operator: {crossover_name} with mutation rate: {mutation_rate} and crossover rate: {crossover_rate}")
                
                for run in range(RUNS):
                    print(f"Run {run + 1}/{RUNS}")
                    _, _, history = genetic_algorithm(
                        groups=groups_list,
                        lessons=lessons_list,
                        rooms=rooms_list,
                        slots=slots_list,
                        population_size=100,
                        generations=250,
                        crossover_rate=crossover_rate,
                        mutation_rate=mutation_rate,
                        fitness_kwargs=default_fitness_params,
                        record_all_generations=True,
                        selection_function=tournament_selection
                    )
                    crossover_histories[variant_name].append(history)

    crossover_table = create_comparison_table(crossover_histories, RUNS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    crossover_table.to_csv(f'crossover_comparison_{timestamp}.csv', index=False)
    print("\nCrossover Comparison Results:")
    print(crossover_table)

    plot_comparison_with_std(
        list(crossover_histories.values()),
        list(crossover_histories.keys()),
        "Comparison of Crossover Operators and Rates (with std dev)",
        "pop100_gen200_crossover_rates_std"
    )

    # Test 2: Compare mutation operators with different rates
    mutation_comparison = {
        "standard": lambda c, rate: mutate(c, rooms_list, slots_list, room_mutation_rate=rate, slot_mutation_rate=rate, validate=True),
        "swap": lambda c, rate: swap_mutation(c, mutation_rate=rate, validate=True),
        "group_reverse": lambda c, rate: group_reverse_mutation(c, mutation_rate=rate, validate=True)
    }

    mutation_histories = {}
    for mutation_name, mutation_func in mutation_comparison.items():
        for mutation_rate in mutation_rates:
            for crossover_rate in crossover_rates:
                variant_name = f"{mutation_name}_{mutation_rate}_crossover_{crossover_rate}"
                mutation_histories[variant_name] = []
                print(f"\nTesting mutation operator: {mutation_name} with mutation rate: {mutation_rate} and crossover rate: {crossover_rate}")
                
                for run in range(RUNS):
                    print(f"Run {run + 1}/{RUNS}")
                    _, _, history = genetic_algorithm(
                        groups=groups_list,
                        lessons=lessons_list,
                        rooms=rooms_list,
                        slots=slots_list,
                        population_size=100,
                        generations=250,
                        crossover_rate=crossover_rate,
                        mutation_rate=mutation_rate,
                        fitness_kwargs=default_fitness_params,
                        record_all_generations=True,
                        selection_function=tournament_selection
                    )
                    mutation_histories[variant_name].append(history)

    mutation_table = create_comparison_table(mutation_histories, RUNS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mutation_table.to_csv(f'mutation_comparison_{timestamp}.csv', index=False)
    print("\nMutation Comparison Results:")
    print(mutation_table)

    plot_comparison_with_std(
        list(mutation_histories.values()),
        list(mutation_histories.keys()),
        "Comparison of Mutation Operators and Rates (with std dev)",
        "pop100_gen200_mutation_rates_comparison_std"
    )

    # Test 3: Compare selection methods (aggregate over all mutation/crossover rate combinations)
    selection_methods = {
        "tournament": tournament_selection,
        "roulette": roulette_wheel_selection,
        "rank": rank_selection,
        "elitist": elitist_selection
    }
    
    selection_histories = {name: [] for name in selection_methods.keys()}
    
    for selection_name, selection_func in selection_methods.items():
        print(f"\nTesting selection method: {selection_name}")
        for mutation_rate in mutation_rates:
            for crossover_rate in crossover_rates:
                combo_desc = f"{selection_name}_{mutation_rate}_crossover_{crossover_rate}"
                print(f"  Combination: mutation_rate={mutation_rate}, crossover_rate={crossover_rate}")
                for run in range(RUNS):
                    print(f"    Run {run + 1}/{RUNS}")
                    _, _, history = genetic_algorithm(
                        groups=groups_list,
                        lessons=lessons_list,
                        rooms=rooms_list,
                        slots=slots_list,
                        population_size=100,
                        generations=250,
                        crossover_rate=crossover_rate,
                        mutation_rate=mutation_rate,
                        fitness_kwargs=default_fitness_params,
                        record_all_generations=True,
                        selection_function=selection_func
                    )
                    selection_histories[selection_name].append(history)
    
    selection_stats = []
    for name, histories in selection_histories.items():
        final_fitnesses = [h[-1] for h in histories] if histories else []
        selection_stats.append({
            "Selection": name,
            "Mean Fitness": np.mean(final_fitnesses) if final_fitnesses else np.nan,
            "Std Dev": np.std(final_fitnesses) if final_fitnesses else np.nan,
            "Max Fitness": np.max(final_fitnesses) if final_fitnesses else np.nan,
            "Min Fitness": np.min(final_fitnesses) if final_fitnesses else np.nan,
            "Runs (total)": len(final_fitnesses)
        })
    
    selection_table = pd.DataFrame(selection_stats)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    selection_table.to_csv(f'selection_comparison_{timestamp}.csv', index=False)
    print("\nSelection Comparison Results:")
    print(selection_table)
    
    plot_comparison_with_std(
        list(selection_histories.values()),
        list(selection_histories.keys()),
        "Comparison of Selection Methods (aggregated over rates)",
        "pop100_gen250_selection_methods_std"
    )

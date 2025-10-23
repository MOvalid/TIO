import os
import itertools
import time
import pandas as pd
import numpy as np
from datetime import datetime
from csv_utils import save_results_to_csv
from data import prepare_data_medium
from genetic_algorithm import genetic_algorithm, tournament_selection
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
        # Convert list of histories to numpy array and handle any NaN/inf values
        histories_array = np.array(histories)
        histories_array = np.nan_to_num(histories_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate mean and std dev with nan handling
        mean_fitness = np.nanmean(histories_array, axis=0)
        std_fitness = np.nanstd(histories_array, axis=0)
        
        # Replace any remaining NaN or inf values
        mean_fitness = np.nan_to_num(mean_fitness, nan=0.0, posinf=0.0, neginf=0.0)
        std_fitness = np.nan_to_num(std_fitness, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Plot mean line
        plt.plot(mean_fitness, label=label)
        # Plot standard deviation area
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
    
    # Ensure diagrams directory exists
    ensure_directory_exists('./diagrams')
    
    # Save in diagrams directory
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
    crossover_comparison = {
        "one_point": one_point_crossover,
        "two_point": two_point_crossover,
        "uniform": lambda p1, p2: uniform_crossover(p1, p2, check_validity=True)
    }

    crossover_histories = {}
    for crossover_name, crossover_func in crossover_comparison.items():
        for rate in crossover_rates:
            variant_name = f"{crossover_name}_{rate}"
            crossover_histories[variant_name] = []
            print(f"\nTesting crossover operator: {crossover_name} with rate: {rate}")
            
            for run in range(RUNS):
                print(f"Run {run + 1}/{RUNS}")
                _, _, history = genetic_algorithm(
                    groups=groups_list,
                    lessons=lessons_list,
                    rooms=rooms_list,
                    slots=slots_list,
                    population_size=100,
                    generations=250,
                    crossover_rate=rate,
                    mutation_rate=0.1,
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
    mutation_rates = [0.05, 0.1, 0.15]
    mutation_comparison = {
        "standard": lambda c, rate: mutate(c, rooms_list, slots_list, room_mutation_rate=rate, slot_mutation_rate=rate, validate=True),
        "swap": lambda c, rate: swap_mutation(c, mutation_rate=rate, validate=True),
        "group_reverse": lambda c, rate: group_reverse_mutation(c, mutation_rate=rate, validate=True)
    }

    mutation_histories = {}
    for mutation_name, mutation_func in mutation_comparison.items():
        for rate in mutation_rates:
            variant_name = f"{mutation_name}_{rate}"
            mutation_histories[variant_name] = []
            print(f"\nTesting mutation operator: {mutation_name} with rate: {rate}")
            
            for run in range(RUNS):
                print(f"Run {run + 1}/{RUNS}")
                _, _, history = genetic_algorithm(
                    groups=groups_list,
                    lessons=lessons_list,
                    rooms=rooms_list,
                    slots=slots_list,
                    population_size=100,
                    generations=250,
                    crossover_rate=0.9,
                    mutation_rate=rate,
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

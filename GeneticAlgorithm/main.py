import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data import prepare_data_medium
from genetic_algorithm import (
    genetic_algorithm,
    tournament_selection
)

def ensure_directory_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_comparison_with_std(histories_list, labels, title, filename_prefix):
    plt.figure(figsize=(12, 8))
    for label, histories in zip(labels, histories_list):
        # histories: list of runs, each is list of fitness by generation
        histories_array = np.array(histories, dtype=float)
        # protect against nan/inf
        histories_array = np.nan_to_num(histories_array, nan=0.0, posinf=0.0, neginf=0.0)
        mean_fitness = np.nanmean(histories_array, axis=0)
        std_fitness = np.nanstd(histories_array, axis=0)
        mean_fitness = np.nan_to_num(mean_fitness, nan=0.0)
        std_fitness = np.nan_to_num(std_fitness, nan=0.0)
        plt.plot(mean_fitness, label=label)
        plt.fill_between(range(len(mean_fitness)),
                         mean_fitness - std_fitness,
                         mean_fitness + std_fitness,
                         alpha=0.2)
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
    return filename

def create_runs_table(final_fitnesses, runtime_seconds):
    df_runs = pd.DataFrame({
        "run": list(range(1, len(final_fitnesses)+1)),
        "final_fitness": final_fitnesses
    })
    summary = {
        "run": "summary",
        "final_fitness": np.nan
    }
    stats = {
        "mean_fitness": np.mean(final_fitnesses),
        "std_fitness": np.std(final_fitnesses),
        "max_fitness": np.max(final_fitnesses),
        "min_fitness": np.min(final_fitnesses),
        "total_runs": len(final_fitnesses),
        "total_time_s": runtime_seconds
    }
    return df_runs, stats

if __name__ == "__main__":
    # Basic GA test for current data (single, configurable experiment repeated RUNS times)
    groups_list, lessons_list, rooms_list, slots_list = prepare_data_medium()

    # User-editable basic configuration
    RUNS = 5
    POPULATION = 100
    GENERATIONS = 250
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.1
    SELECTION = tournament_selection  # basic selection

    default_fitness_params = {
        "init_score": 0,

        # "group_conflict_penalty": 300,
        # "teacher_conflict_penalty": 200,
        # "room_conflict_penalty": 150,

        # "group_gap_penalty_short": 10,
        # "group_gap_penalty_long": 20,
        # "teacher_gap_penalty_short": 5,
        # "teacher_gap_penalty_long": 10,

        # "concentration_bonus": 30,


        "group_conflict_penalty": 150,
        "teacher_conflict_penalty": 120,
        "room_conflict_penalty": 100,

        "group_gap_penalty_short": 5,
        "group_gap_penalty_long": 10,
        "teacher_gap_penalty_short": 2,
        "teacher_gap_penalty_long": 5,

        "concentration_bonus": 30,
    }

    ensure_directory_exists('./diagrams')
    ensure_directory_exists('./results')

    # Run GA multiple times with same basic params
    all_histories = []
    final_fitnesses = []
    start_all = time.time()
    for run in range(1, RUNS + 1):
        print(f"Basic GA run {run}/{RUNS} — pop={POPULATION} gen={GENERATIONS} cr={CROSSOVER_RATE} mr={MUTATION_RATE}")
        t0 = time.time()
        best_chrom, best_gen, history = genetic_algorithm(
            groups=groups_list,
            lessons=lessons_list,
            rooms=rooms_list,
            slots=slots_list,
            population_size=POPULATION,
            generations=GENERATIONS,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            fitness_kwargs=default_fitness_params,
            record_all_generations=True,
            selection_function=SELECTION,
            elite_size=0
        )
        t1 = time.time()
        run_time = t1 - t0
        history = history or []
        all_histories.append(history)
        final = history[-1] if len(history) > 0 else np.nan
        final_fitnesses.append(final)
        print(f"  Run time: {run_time:.2f}s final fitness: {final:.4f}")

    total_time = time.time() - start_all

    # Save per-run results + summary
    df_runs, stats = create_runs_table(final_fitnesses, total_time)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_csv = f"./results/basic_ga_runs_{timestamp}.csv"
    df_runs.to_csv(runs_csv, index=False)

    summary_csv = f"./results/basic_ga_summary_{timestamp}.csv"
    pd.DataFrame([stats]).to_csv(summary_csv, index=False)

    # Plot mean + std across runs (single-line grouped)
    label = f"basicGA_pop{POPULATION}_gen{GENERATIONS}_cr{CROSSOVER_RATE}_mr{MUTATION_RATE}"
    png = plot_comparison_with_std([all_histories], [label],
                                   "Basic GA performance (mean ± std)", f"{label}")

    print(f"Saved runs CSV: {runs_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved plot: {png}")

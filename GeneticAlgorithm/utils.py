import os
import datetime
import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List, Tuple, Dict, Any, Optional
from models import Chromosome
from collections import defaultdict


DAYS_ORDER = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


def display_schedule(chromosome: Chromosome, sort: bool = True) -> None:
    if sort:
        chromosome = sorted(
            chromosome,
            key=lambda g: (DAYS_ORDER.get(g.slot.day, 7), g.slot.start_time, g.group)
        )
    
    print(f"{'Day':<3} | {'Time':<11} | {'Group':<6} | {'Teacher':<10} | {'Room':<5} | {'Lesson':<20}")
    print("-" * 80)
    
    for gene in chromosome:
        print(f"{gene.slot.day:<3} | "
              f"{gene.slot.start_time}-{gene.slot.end_time:<5} | "
              f"{gene.group:<6} | "
              f"{gene.lesson.teacher:<10} | "
              f"{gene.room:<5} | "
              f"{gene.lesson.name:<20}")


def display_schedule_table(chromosome: Chromosome) -> None:
    days = sorted(set(gene.slot.day for gene in chromosome), key=lambda d: DAYS_ORDER.get(d, 7))
    hours = sorted(set(gene.slot.start_time for gene in chromosome))

    table = pd.DataFrame("", index=hours, columns=days)

    for gene in chromosome:
        content = f"{gene.group}: {gene.lesson.name} ({gene.room})"
        table.at[gene.slot.start_time, gene.slot.day] += content + "\n"

    print(table)


def show_schedule_gui(chromosome: Chromosome) -> None:
    days = sorted(set(g.slot.day for g in chromosome), key=lambda d: DAYS_ORDER.get(d, 7))
    hours = sorted(set(g.slot.start_time for g in chromosome))

    root = tk.Tk()
    root.title("Timetable Schedule")

    for j, day in enumerate(days):
        tk.Label(root, text=day, borderwidth=1, relief="solid", width=20, bg="lightblue").grid(row=0, column=j+1)

    for i, hour in enumerate(hours):
        tk.Label(root, text=hour, borderwidth=1, relief="solid", width=10, bg="lightgreen").grid(row=i+1, column=0)

    for i, hour in enumerate(hours):
        for j, day in enumerate(days):
            cell_text = ""
            for gene in chromosome:
                if gene.slot.day == day and gene.slot.start_time == hour:
                    cell_text += f"{gene.group}: {gene.lesson.name} [{gene.lesson.teacher}] ({gene.room})\n"
            tk.Label(
                root, text=cell_text, borderwidth=1, relief="solid",
                width=25, height=6, justify="left", anchor="nw"
            ).grid(row=i+1, column=j+1, sticky="nsew")

    root.mainloop()


def print_best_config(best_score: int, best_config: Tuple[Dict[str, Any], Dict[str, Any], float]) -> None:
    f_params, ga_params, duration = best_config

    print("\n=== BEST OVERALL SOLUTION ===")
    print(f"Best fitness: {best_score}")

    print("\nBest fitness params:")
    for k, v in f_params.items():
        print(f"  {k:<25}: {v}")

    print("\nBest GA params:")
    for k, v in ga_params.items():
        print(f"  {k:<25}: {v}")

    print(f"\nExecution time: {duration:.2f} seconds\n")


def summarize_ga_results(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    df['fitness'] = df['fitness'].replace({'-inf': float('-inf')})

    group_cols = ['population_size', 'generations', 'crossover_rate', 'mutation_rate', 'patience']

    grouped = df.groupby(group_cols)

    summary_rows = []
    for name, group in grouped:
        total_runs = len(group)
        invalid_runs = (group['fitness'] == float('-inf')).sum()
        valid_fitnesses = group.loc[group['fitness'] != float('-inf'), 'fitness']
        avg_fitness = valid_fitnesses.mean() if not valid_fitnesses.empty else None
        valid_times = group.loc[group['fitness'] != float('-inf'), 'time']
        avg_time = valid_times.mean() if not valid_times.empty else None
        
        most_common_generation = group['generations'].mode().iloc[0] if not group['generations'].mode().empty else None
        
        summary_rows.append({
            'Population Size': name[0],
            'Generations': name[1],
            'Crossover Rate': name[2],
            'Mutation Rate': name[3],
            'Patience': name[4],
            'Total Runs': total_runs,
            'Invalid Runs (-inf)': invalid_runs,
            'Avg Fitness': f"{avg_fitness:.2f}" if avg_fitness is not None else 'N/A',
            'Avg Time (s)': f"{avg_time:.2f}" if avg_time is not None else 'N/A',
            'Most Common Generations': most_common_generation
        })

    summary_df = pd.DataFrame(summary_rows)

    print(summary_df.to_string(index=False))


def plot_fitness_progression(fitness_history: List[int], title: str = "Fitness Progression Over Generations") -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_history, label="Best Fitness", color='blue')
    plt.xlabel("Generations")
    plt.ylabel("Best Fitness")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def analyze_and_plot_best_repeat(fitness_matrix: np.ndarray, plot_func, title_prefix: str = "Best Fitness Progression") -> None:
    max_fitness_vals = np.array([max(fit_list) for fit_list in fitness_matrix])
    best_repeat_index = np.argmax(max_fitness_vals)
    best_run = fitness_matrix[best_repeat_index]

    running_best = []
    current_best = -np.inf
    for fitness in best_run:
        if fitness > current_best:
            current_best = fitness
        running_best.append(current_best)

    best_iter_index = np.argmax(running_best)

    print(f"\nBest fitness found in repeat #{best_repeat_index+1} at iteration {best_iter_index} "
          f"with fitness {running_best[best_iter_index]}")

    plot_func(
        running_best,
        title=f"{title_prefix} (Repeat #{best_repeat_index+1})"
    )



######=== Multiwindow results plot generator #######

def prepare_label(params: Dict[str, Any], exclude_keys: List[str]) -> str:
    label_parts = []
    for k, v in params.items():
        if k not in exclude_keys:
            if k == "patience" and v is None:
                continue
            label_parts.append(f"{k}={v}")
    return ", ".join(label_parts) if label_parts else "default params"

def plot_single_curve(ax: Axes, fitness_history: List[float], label: str) -> None:
    running_best = []
    current_best = -float('inf')
    for f in fitness_history:
        if f > current_best:
            current_best = f
        running_best.append(current_best)
    ax.plot(running_best, label=label)

def generate_figure_without_crossover(pop_size: int, generations: int, entries: List[Tuple[Dict[str, Any], Optional[List[float]]]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    for params, fitness_history in entries:
        if fitness_history is None:
            continue
        label = prepare_label(params, exclude_keys=["population_size", "generations"])
        plot_single_curve(ax, fitness_history, label)

    ax.set_title(f"Best Fitness Progression\nPopulation Size: {pop_size}, Generations: {generations}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness so far")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    ax.grid(True)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    return fig

def generate_figure_with_crossover(pop_size: int, generations: int, entries: List[Tuple[Dict[str, Any], Optional[List[float]]]]) -> plt.Figure:
    crossover_groups = defaultdict(list)
    for params, fitness_history in entries:
        crossover = params.get("crossover_rate", "unknown")
        crossover_groups[crossover].append((params, fitness_history))

    crossover_values = sorted(crossover_groups.keys())
    n_subplots = len(crossover_values)

    fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 5 * n_subplots), sharex=True)
    if n_subplots == 1:
        axes = [axes]

    for ax, crossover in zip(axes, crossover_values):
        ax.set_title(f"Crossover rate: {crossover}")
        for params, fitness_history in crossover_groups[crossover]:
            if fitness_history is None:
                continue
            label = prepare_label(params, exclude_keys=["population_size", "generations", "crossover_rate"])
            plot_single_curve(ax, fitness_history, label)

        ax.set_ylabel("Best Fitness so far")
        ax.grid(True)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    axes[-1].set_xlabel("Generation")
    fig.suptitle(f"Population Size: {pop_size}, Generations: {generations}")
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    return fig

def display_or_save_figure(fig: plt.Figure, save_path: Optional[str] = None) -> None:
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        fig.show()

def get_timestamped_filepath(directory: str, base_filename: str, extension: str = ".png") -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_filename}_{timestamp}{extension}"
    return os.path.join(directory, filename)

def plot_fitness_split_by_crossover_and_pop_gen(
    fitness_histories: Dict[Tuple[str, ...], Optional[List[float]]],
    split_by_crossover: bool = True,
    save_dir: Optional[str] = r"./diagrams"
) -> None:
    pop_gen_groups = defaultdict(list)

    for params_tuple, fitness_history in fitness_histories.items():
        params = dict(params_tuple)
        key = (params["population_size"], params["generations"])
        pop_gen_groups[key].append((params, fitness_history))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for (pop_size, generations), entries in pop_gen_groups.items():
        if not split_by_crossover:
            fig = generate_figure_without_crossover(pop_size, generations, entries)
            save_path = None
            if save_dir:
                base_name = f"pop{pop_size}_gen{generations}"
                save_path = get_timestamped_filepath(save_dir, base_name)
            display_or_save_figure(fig, save_path)
        else:
            fig = generate_figure_with_crossover(pop_size, generations, entries)
            save_path = None
            if save_dir:
                base_name = f"pop{pop_size}_gen{generations}_crossover"
                save_path = get_timestamped_filepath(save_dir, base_name)
            display_or_save_figure(fig, save_path)

import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from models import AllResults, BestParams, BestStats, Chromosome
from datetime import datetime

sns.set(style="whitegrid")
RESULTS_DIR = rf".\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_plot_with_timestamp(plt_object: plt, prefix: str) -> str:
    """
    Saves the given matplotlib plot object with a timestamp in the RESULTS_DIR.
    
    Args:
        plt_object: The matplotlib.pyplot object containing the plot.
        prefix: Prefix for the filename (e.g., 'fitness_progress', 'timetable').
        
    Returns:
        The full path of the saved file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RESULTS_DIR, f"{prefix}_{timestamp}.png")
    plt_object.savefig(filename)
    print(f"Saved plot to {filename}")
    return filename

def plot_fitness_progress(stats: BestStats,
                          title: str = "Fitness Progress Over Iterations",
                          best_params: BestParams | None = None,
                          show: bool = True):
    """
    Plots the best and mean fitness per iteration.
    Optionally includes the best parameters in the title.
    
    Args:
        stats: List of dictionaries containing iteration stats ('iteration', 'best_score', 'mean_score').
        title: Plot title.
        best_params: Optional tuple of (n_ants, n_iterations, evaporation_rate, alpha, beta) to include in the title.
        show: Whether to display the plot interactively.
    """
    iterations = [s['iteration'] for s in stats]
    best_scores = [s['best_score'] for s in stats]
    mean_scores = [s['mean_score'] for s in stats]

    if best_params is not None:
        params_str = f"n_ants={best_params[0]}, n_iter={best_params[1]}, evap={best_params[2]}, alpha={best_params[3]}, beta={best_params[4]}"
        title = f"{title} ({params_str})"

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, best_scores, label='Best Fitness', color='blue', marker='o')
    plt.plot(iterations, mean_scores, label='Mean Fitness', color='orange', marker='x')
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    save_plot_with_timestamp(plt, "fitness_progress")
    if show:
        plt.show()
    plt.close()


def compare_parameter_sets(results_list: AllResults, param_names: List[str], metric: str = 'best_score', show: bool = True):
    """
    Compares multiple parameter sets by plotting the specified metric.
    
    Args:
        results_list: List of dictionaries containing 'params' and metric values.
        param_names: List of parameter names (e.g., ['n_ants', 'n_iterations', 'evaporation_rate']).
        metric: The metric to plot (default 'best_score').
    """
    labels = [str(res['params']) for res in results_list]
    scores = [res[metric] for res in results_list]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=labels, y=scores, palette="viridis")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(metric.replace("_", " ").title())
    plt.xlabel("Parameter Set")
    plt.title(f"{metric.replace('_', ' ').title()} Across Parameter Sets")
    plt.tight_layout()

    save_plot_with_timestamp(plt, "parameter_comparison")
    if show:
        plt.show()
    plt.close()


def plot_timetable(chromosome: Chromosome, title: str = "Timetable", show: bool = True):
    """
    Displays a heatmap-style timetable for groups vs. time slots.
    
    Args:
        chromosome: The chromosome representing the timetable.
        title: Plot title.
    """
    groups = sorted({gene.group for gene in chromosome})
    slots = sorted({f"{gene.slot.day} {gene.slot.start_time}" for gene in chromosome})

    table = {group: {slot: "" for slot in slots} for group in groups}

    for gene in chromosome:
        slot_key = f"{gene.slot.day} {gene.slot.start_time}"
        table[gene.group][slot_key] = gene.lesson.name

    data_matrix = [[table[group][slot] for slot in slots] for group in groups]

    plt.figure(figsize=(12, len(groups) * 0.5 + 2))
    sns.heatmap([[1 if cell else 0 for cell in row] for row in data_matrix],
                annot=data_matrix, fmt="", cmap="Pastel1",
                cbar=False, linewidths=0.5, linecolor="gray")
    plt.yticks(range(len(groups)), groups, rotation=0)
    plt.xticks(range(len(slots)), slots, rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    
    save_plot_with_timestamp(plt, "timetable")
    if show:
        plt.show()
    plt.close()

import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from models import Chromosome

sns.set(style="whitegrid")


def plot_fitness_progress(stats: List[Dict[str, float]], title: str = "Fitness Progress Over Iterations"):
    """
    Plots the best and mean fitness per iteration.
    
    Args:
        stats: List of dictionaries containing iteration stats ('iteration', 'best_score', 'mean_score').
        title: Plot title.
    """
    iterations = [s['iteration'] for s in stats]
    best_scores = [s['best_score'] for s in stats]
    mean_scores = [s['mean_score'] for s in stats]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, best_scores, label='Best Fitness', color='blue', marker='o')
    plt.plot(iterations, mean_scores, label='Mean Fitness', color='orange', marker='x')
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_parameter_sets(results_list: List[Dict], param_names: List[str], metric: str = 'best_score'):
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
    plt.show()


def plot_timetable(chromosome: Chromosome, title: str = "Timetable"):
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
    plt.show()

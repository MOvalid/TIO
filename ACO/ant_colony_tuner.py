import itertools
from typing import List, Tuple, Dict, Any
from ant_colony_planner import AntColonyPlanner
from data import prepare_data_medium
from display_utils import display_timetable
from fitness_utils import calculate_fitness
from models import Chromosome, Lesson, Slot


def run_aco(groups: List[str], lessons: List[Lesson], rooms: List[str], slots: List[Slot],
            n_ants: int, n_iterations: int, evaporation_rate: float) -> Dict[str, Any]:
    """
    Runs the Ant Colony Optimization algorithm with specified parameters.
    
    Returns:
        Dictionary containing best_solution, best_score, and iteration_stats.
    """
    aco = AntColonyPlanner(
        groups, lessons, rooms, slots,
        fitness_function=calculate_fitness,
        n_ants=n_ants,
        n_iterations=n_iterations,
        evaporation_rate=evaporation_rate
    )
    return aco.run()


def tune_parameters(groups: List[str], lessons: List[Lesson], rooms: List[str], slots: List[Slot],
                    n_ants_options: List[int], n_iterations_options: List[int], evaporation_options: List[float]
                   ) -> Tuple[Chromosome, float, Tuple[int, int, float], List[Dict[str, float]]]:
    """
    Tests all combinations of parameters and returns the best solution and its parameters.
    
    Returns:
        best_plan: Best chromosome found
        best_score: Fitness score of best_plan
        best_params: Tuple of (n_ants, n_iterations, evaporation_rate)
        best_stats: Iteration stats of best run
    """
    best_overall_score = float("-inf")
    best_plan = None
    best_params = None
    best_stats = None

    for n_ants, n_iterations, evaporation_rate in itertools.product(
        n_ants_options, n_iterations_options, evaporation_options
    ):
        print(f"Testing: n_ants={n_ants}, n_iterations={n_iterations}, evaporation_rate={evaporation_rate}")
        results = run_aco(groups, lessons, rooms, slots, n_ants, n_iterations, evaporation_rate)
        score = results["best_score"]

        if score > best_overall_score:
            best_overall_score = score
            best_plan = results["best_solution"]
            best_params = (n_ants, n_iterations, evaporation_rate)
            best_stats = results["iteration_stats"]

    return best_plan, best_overall_score, best_params, best_stats


def display_results(best_plan: Chromosome, best_score: float, best_params: Tuple[int, int, float], stats: List[Dict[str, float]]):
    """
    Displays the best timetable, its score, parameters, and iteration statistics.
    """
    print("\n=== Best Parameter Combination ===")
    print(f"n_ants={best_params[0]}, n_iterations={best_params[1]}, evaporation_rate={best_params[2]}")
    print(f"Best fitness: {best_score}\n")

    display_timetable(best_plan)

    print("\nIteration stats for best configuration:")
    for stat in stats:
        print(f"Iteration {stat['iteration']}: best = {stat['best_score']}, mean = {stat['mean_score']:.2f}")

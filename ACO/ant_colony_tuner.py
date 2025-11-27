import itertools
from typing import List, Tuple, Dict, Any
from ant_colony_planner import AntColonyPlanner
from display_utils import display_timetable
from fitness_utils import calculate_fitness
from models import AllResults, BestParams, BestStats, Chromosome, Lesson, Slot



def run_aco(groups: List[str], lessons: List[Lesson], rooms: List[str], slots: List[Slot],
            n_ants: int, n_iterations: int, evaporation_rate: float,
            alpha: float = 1.0, beta: float = 2.0
    ) -> Dict[str, Any]:
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
        evaporation_rate=evaporation_rate,
        alpha=alpha,
        beta=beta,
        local_evaporation_rate=evaporation_rate
    )
    return aco.run()


def tune_parameters(groups: List[str], lessons: List[Lesson], rooms: List[str], slots: List[Slot],
                    n_ants_options: List[int], n_iterations_options: List[int], evaporation_options: List[float],
                    alpha_options: List[float], beta_options: List[float]
    ) -> Tuple[Chromosome, float, BestParams, BestStats, AllResults]:
    """
    Tests all combinations of parameters and returns the best solution and its parameters.
    
    Returns:
        best_plan: Best chromosome found
        best_score: Fitness score of best_plan
        best_params: Tuple of (n_ants, n_iterations, evaporation_rate, alpha, beta)
        best_stats: Iteration stats of best run
        all_results: List of dictionaries with keys 'params', 'best_score', 'mean_score', 'iteration_stats'
    """
    best_overall_score = float("-inf")
    best_plan = None
    best_params = None
    best_stats = None
    all_results = []

    for n_ants, n_iterations, evaporation_rate, alpha, beta in itertools.product(
        n_ants_options, n_iterations_options, evaporation_options, alpha_options, beta_options
    ):
        print(f"Testing: n_ants={n_ants}, n_iterations={n_iterations}, evaporation_rate={evaporation_rate}, alpha={alpha}, beta={beta}")
        results = run_aco(groups, lessons, rooms, slots, n_ants, n_iterations, evaporation_rate, alpha, beta)
        score = results["best_score"]
        mean_score = sum([s["mean_score"] for s in results["iteration_stats"]]) / len(results["iteration_stats"])

        all_results.append({
            "params": (n_ants, n_iterations, evaporation_rate, alpha, beta),
            "best_score": score,
            "mean_score": mean_score,
            "iteration_stats": results["iteration_stats"]
        })

        if score > best_overall_score:
            best_overall_score = score
            best_plan = results["best_solution"]
            best_params = (n_ants, n_iterations, evaporation_rate, alpha, beta)
            best_stats = results["iteration_stats"]

    return best_plan, best_overall_score, best_params, best_stats, all_results


def display_results(best_plan: Chromosome, best_score: float, best_params: BestParams, stats: BestStats):
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

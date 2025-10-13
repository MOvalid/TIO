import random
import time
import numpy as np
from typing import List, Dict, Tuple, Any
from csv_utils import export_result_incremental
from models import Gen, Chromosome, Lesson, Slot
from genetic_utils import generate_random_chromosome, calculate_fitness
from genetic_operators import mutate, one_point_crossover
from utils import plot_fitness_progression

CSV_FILENAME = "grid_search_results.csv"

def tournament_selection(
    population: List[Chromosome],
    k: int = 2,
    fitness_kwargs: dict = {}
) -> Chromosome:
    selected: List[Chromosome] = random.sample(population, k)
    selected.sort(key=lambda chromo: calculate_fitness(chromo, **fitness_kwargs), reverse=True)
    return selected[0]


def genetic_algorithm(
    groups: List[str],
    lessons: List[Lesson],
    rooms: List[str],
    slots: List[Slot],
    population_size: int = 20,
    generations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    patience: int = None,
    fitness_kwargs: dict = {},
    record_all_generations: bool = False
) -> Tuple[Chromosome, int, List[int]]:

    population: List[Chromosome] = [
        generate_random_chromosome(groups, lessons, rooms, slots)
        for _ in range(population_size)
    ]

    best_chromosome: Chromosome = population[0]
    best_fitness: int = calculate_fitness(best_chromosome, **fitness_kwargs)
    fitness_history: List[int] = [best_fitness]

    no_improvement: int = 0

    for gen in range(generations):
        new_population: List[Chromosome] = []

        while len(new_population) < population_size:
            parent1: Chromosome = tournament_selection(population, fitness_kwargs=fitness_kwargs)
            parent2: Chromosome = tournament_selection(population, fitness_kwargs=fitness_kwargs)

            if random.random() < crossover_rate:
                child1, child2 = one_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            child1 = mutate(child1, rooms, slots, mutation_rate, mutation_rate)
            child2 = mutate(child2, rooms, slots, mutation_rate, mutation_rate)

            new_population.extend([child1, child2])

        population = new_population[:population_size]

        current_best_fitness = best_fitness
        improved = False

        gen_best_chromo = max(population, key=lambda chromo: calculate_fitness(chromo, **fitness_kwargs))
        gen_best_fitness = calculate_fitness(gen_best_chromo, **fitness_kwargs)

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_chromosome = gen_best_chromo
            improved = True

        if record_all_generations:
            fitness_history.append(gen_best_fitness)
        else:
            fitness_history.append(best_fitness)

        if improved:
            no_improvement = 0
        else:
            no_improvement += 1

        if patience is not None and best_fitness != float('-inf') and no_improvement >= patience:
            print(f"Early stopping at generation {gen}, best fitness = {best_fitness}")
            break

    return best_chromosome, best_fitness, fitness_history


def grid_search_ga(
    groups: List[str],
    lessons: List[Lesson],
    rooms: List[str],
    slots: List[Slot],
    fitness_grid: List[Dict[str, Any]],
    ga_params_grid: List[Dict[str, Any]],
    record_all_generations: bool = False
) -> Tuple[Chromosome, int, Tuple[Dict[str, Any], Dict[str, Any], float], List[Dict[str, Any]]]:

    best_solution: Chromosome = []
    best_score: int = float("-inf")
    best_config: Tuple[Dict[str, Any], Dict[str, Any], float] = ({}, {}, 0.0)
    results: List[Dict[str, Any]] = []

    total_runs = len(fitness_grid) * len(ga_params_grid)
    run_counter = 1

    best_run_history = []

    for f_params in fitness_grid:
        for ga_params in ga_params_grid:
            print(f"\n=== Run {run_counter}/{total_runs} ===")
            print("Fitness params:", f_params)
            print("GA params:", ga_params)

            start = time.time()
            best_chromosome, best_fitness, fitness_history = genetic_algorithm(
                groups=groups,
                lessons=lessons,
                rooms=rooms,
                slots=slots,
                fitness_kwargs=f_params,
                record_all_generations=record_all_generations,
                **ga_params
            )
            duration = time.time() - start

            print(f"Result fitness: {best_fitness}, time: {duration:.2f}s")

            result_entry = {
                "fitness_params": f_params,
                "ga_params": ga_params,
                "fitness": best_fitness,
                "time": duration
            }
            results.append(result_entry)
            export_result_incremental(result_entry, CSV_FILENAME)

            if best_fitness > best_score:
                best_score = best_fitness
                best_solution = best_chromosome
                best_config = (f_params, ga_params, duration)
                best_run_history = fitness_history

            run_counter += 1

    plot_fitness_progression(best_run_history, "Fitness Progression of the Best Run")

    return best_solution, best_score, best_config, results

def repeat_best_config(
    repeats: int,
    groups_list,
    lessons_list,
    rooms_list,
    slots_list,
    best_configuration,
    genetic_algorithm_func
) -> np.ndarray:
    f_params, ga_params, _ = best_configuration
    best_fitnesses_per_iteration = []

    for i in range(repeats):
        print(f"\n=== Repeat run {i+1}/{repeats} with best config ===")
        _, _, fitness_history = genetic_algorithm_func(
            groups=groups_list,
            lessons=lessons_list,
            rooms=rooms_list,
            slots=slots_list,
            fitness_kwargs=f_params,
            record_all_generations=True,
            **ga_params
        )
        best_fitnesses_per_iteration.append(fitness_history)

    return np.array(best_fitnesses_per_iteration, dtype=object)

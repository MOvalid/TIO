import random
import time
from typing import List, Dict, Tuple, Any
from csv_utils import export_result_incremental
from models import Gen, Chromosome, Lesson, Slot
from genetic_utils import generate_random_chromosome, calculate_fitness
from genetic_operators import mutate, one_point_crossover

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
    patience: int = 100,
    fitness_kwargs: dict = {}
) -> Tuple[Chromosome, int]:

    population: List[Chromosome] = [
        generate_random_chromosome(groups, lessons, rooms, slots)
        for _ in range(population_size)
    ]

    best_chromosome: Chromosome = population[0]
    best_fitness: int = calculate_fitness(best_chromosome, **fitness_kwargs)

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

            child1 = mutate(child1, rooms, slots, mutation_rate)
            child2 = mutate(child2, rooms, slots, mutation_rate)

            new_population.extend([child1, child2])

        population = new_population[:population_size]

        improved = False
        for chromo in population:
            fitness: int = calculate_fitness(chromo, **fitness_kwargs)
            if fitness > best_fitness:
                best_chromosome = chromo
                best_fitness = fitness
                improved = True

        if improved:
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f"Early stopping at generation {gen}, best fitness = {best_fitness}")
            break

    return best_chromosome, best_fitness


def grid_search_ga(
    groups: List[str],
    lessons: List[Lesson],
    rooms: List[str],
    slots: List[Slot],
    fitness_grid: List[Dict[str, Any]],
    ga_params_grid: List[Dict[str, Any]]
) -> Tuple[Chromosome, int, Tuple[Dict[str, Any], Dict[str, Any], float], List[Dict[str, Any]]]:

    best_solution: Chromosome = []
    best_score: int = float("-inf")
    best_config: Tuple[Dict[str, Any], Dict[str, Any], float] = ({}, {}, 0.0)
    results: List[Dict[str, Any]] = []

    total_runs: int = len(fitness_grid) * len(ga_params_grid)
    run_counter: int = 1

    for f_params in fitness_grid:
        for ga_params in ga_params_grid:
            print(f"\n=== Run {run_counter}/{total_runs} ===")
            print("Fitness params:", f_params)
            print("GA params:", ga_params)

            start = time.time()
            best_chromosome, best_fitness = genetic_algorithm(
                groups=groups,
                lessons=lessons,
                rooms=rooms,
                slots=slots,
                fitness_kwargs=f_params,
                **ga_params
            )
            duration: float = time.time() - start

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

            run_counter += 1

    return best_solution, best_score, best_config, results

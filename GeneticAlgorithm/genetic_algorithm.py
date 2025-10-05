import random
from typing import List, Tuple
from models import Gen, Chromosome, Lesson, Slot
from genetic_utils import generate_random_chromosome, calculate_fitness
from genetic_operators import mutate, one_point_crossover

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
    fitness_kwargs: dict = {}
) -> Tuple[Chromosome, int]:

    population: List[Chromosome] = [
        generate_random_chromosome(groups, lessons, rooms, slots)
        for _ in range(population_size)
    ]

    best_chromosome: Chromosome = population[0]
    best_fitness: int = calculate_fitness(best_chromosome, **fitness_kwargs)

    for gen in range(generations):
        new_population: List[Chromosome] = []

        while len(new_population) < population_size:
            # Selection
            parent1: Chromosome = tournament_selection(population, fitness_kwargs=fitness_kwargs)
            parent2: Chromosome = tournament_selection(population, fitness_kwargs=fitness_kwargs)

            # Crossover
            if random.random() < crossover_rate:
                child1, child2 = one_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            # Mutation
            child1 = mutate(child1, rooms, slots, mutation_rate)
            child2 = mutate(child2, rooms, slots, mutation_rate)

            new_population.extend([child1, child2])

        population = new_population[:population_size]

        for chromo in population:
            fitness: int = calculate_fitness(chromo, **fitness_kwargs)
            if fitness > best_fitness:
                best_chromosome = chromo
                best_fitness = fitness

    return best_chromosome, best_fitness

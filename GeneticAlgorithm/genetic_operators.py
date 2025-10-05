import random
from typing import List, Tuple
from models import Gen, Chromosome, Lesson, Slot

def mutate(
    chromosome: Chromosome,
    rooms: List[str],
    slots: List[Slot],
    room_mutation_rate: float = 0.1,
    slot_mutation_rate: float = 0.1
) -> Chromosome:
    
    new_chromosome: Chromosome = []

    for gene in chromosome:
        new_gene = Gen(
            group=gene.group,
            lesson=gene.lesson,
            room=gene.room,
            slot=gene.slot
        )

        # Mutate room
        if random.random() < room_mutation_rate:
            new_gene.room = random.choice(rooms)

        # Mutate slot
        if random.random() < slot_mutation_rate:
            new_gene.slot = random.choice(slots)

        new_chromosome.append(new_gene)
    
    return new_chromosome


def one_point_crossover(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
    if len(parent1) != len(parent2):
        raise ValueError("Parents must be of the same length for crossover.")

    point: int = random.randint(1, len(parent1) - 1)
    
    child1: Chromosome = parent1[:point] + parent2[point:]
    child2: Chromosome = parent2[:point] + parent1[point:]
    
    return child1, child2

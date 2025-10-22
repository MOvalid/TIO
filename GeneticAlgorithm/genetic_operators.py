import random
from typing import Dict, List, Tuple
from genetic_utils import is_valid_chromosome
from models import Gen, Chromosome, Slot
from collections import defaultdict

def mutate(
    chromosome: Chromosome,
    rooms: List[str],
    slots: List[Slot],
    room_mutation_rate: float = 0.1,
    slot_mutation_rate: float = 0.1,
    validate: bool = True,
    max_attempts: int = 100
) -> Chromosome:
    original_chromosome = chromosome[:]
    attempts = 0
    
    while attempts < max_attempts:
        new_chromosome: Chromosome = []

        for gene in chromosome:
            new_gene = Gen(
                group=gene.group,
                lesson=gene.lesson,
                room=gene.room,
                slot=gene.slot
            )

            if random.random() < room_mutation_rate:
                new_gene.room = random.choice(rooms)
            if random.random() < slot_mutation_rate:
                new_gene.slot = random.choice(slots)

            new_chromosome.append(new_gene)
        
        if not validate or is_valid_chromosome(new_chromosome):
            return new_chromosome
            
        attempts += 1
    
    print(f"Warning: Could not find valid mutation after {max_attempts} attempts")
    return original_chromosome


def one_point_crossover(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
    if len(parent1) != len(parent2):
        raise ValueError("Parents must be of the same length for crossover.")

    point: int = random.randint(1, len(parent1) - 1)
    print(f"Crossover point: {point}")
    
    child1: Chromosome = parent1[:point] + parent2[point:]
    child2: Chromosome = parent2[:point] + parent1[point:]
    
    return child1, child2


def two_point_crossover(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
    if len(parent1) != len(parent2):
        raise ValueError("Parents must be of the same length for crossover.")

    length = len(parent1)
    point1 = random.randint(1, length - 2)
    print(f"Crossover point 1: {point1}")
    point2 = random.randint(point1 + 1, length - 1)
    print(f"Crossover point 2: {point2}")
    
    child1: Chromosome = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2: Chromosome = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    
    return child1, child2


def uniform_crossover(
    parent1: Chromosome,
    parent2: Chromosome,
    swap_prob: float = 0.5,
    check_validity: bool = True
) -> Tuple[Chromosome, Chromosome]:
    if len(parent1) != len(parent2):
        raise ValueError("Parents must be of the same length for crossover.")

    child1: Chromosome = parent1.copy()
    child2: Chromosome = parent2.copy()
    
    if check_validity:
        def get_used_slots(chromosome: Chromosome) -> Tuple[Dict, Dict, Dict]:
            group_slots = defaultdict(set)
            teacher_slots = defaultdict(set)
            room_slots = defaultdict(set)
            
            for gene in chromosome:
                time_key = (gene.slot.day, gene.slot.start_time)
                group_slots[gene.group].add(time_key)
                teacher_slots[gene.lesson.teacher].add(time_key)
                room_slots[gene.room].add(time_key)
                
            return group_slots, teacher_slots, room_slots
        
        def can_swap(gene1: Gen, gene2: Gen, c1_slots: Tuple[Dict, Dict, Dict], c2_slots: Tuple[Dict, Dict, Dict]) -> bool:
            time_key1 = (gene1.slot.day, gene1.slot.start_time)
            time_key2 = (gene2.slot.day, gene2.slot.start_time)
            
            # Remove current slots from tracking
            c1_group, c1_teacher, c1_room = c1_slots
            c2_group, c2_teacher, c2_room = c2_slots
            
            c1_group[gene1.group].remove(time_key1)
            c1_teacher[gene1.lesson.teacher].remove(time_key1)
            c1_room[gene1.room].remove(time_key1)
            
            c2_group[gene2.group].remove(time_key2)
            c2_teacher[gene2.lesson.teacher].remove(time_key2)
            c2_room[gene2.room].remove(time_key2)
            
            valid1 = (time_key2 not in c1_group[gene2.group] and 
                     time_key2 not in c1_teacher[gene2.lesson.teacher] and 
                     time_key2 not in c1_room[gene2.room])
            
            valid2 = (time_key1 not in c2_group[gene1.group] and 
                     time_key1 not in c2_teacher[gene1.lesson.teacher] and 
                     time_key1 not in c2_room[gene1.room])

            c1_group[gene1.group].add(time_key1)
            c1_teacher[gene1.lesson.teacher].add(time_key1)
            c1_room[gene1.room].add(time_key1)
            
            c2_group[gene2.group].add(time_key2)
            c2_teacher[gene2.lesson.teacher].add(time_key2)
            c2_room[gene2.room].add(time_key2)
            
            return valid1 and valid2

        child1_slots = get_used_slots(child1)
        child2_slots = get_used_slots(child2)
        
        # Try to swap genes with validity check
        for i in range(len(child1)):
            if random.random() < swap_prob:
                if can_swap(child1[i], child2[i], child1_slots, child2_slots):
                    child1[i], child2[i] = child2[i], child1[i]
    else:
        # Simple uniform crossover without validity checks
        for i in range(len(child1)):
            if random.random() < swap_prob:
                child1[i], child2[i] = child2[i], child1[i]
    
    return child1, child2

def swap_mutation(
    chromosome: Chromosome,
    mutation_rate: float = 0.1,
    validate: bool = True,
    max_attempts: int = 100
) -> Chromosome:
    original_chromosome = chromosome[:]
    attempts = 0
    
    while attempts < max_attempts:
        new_chromosome = chromosome[:]
        length = len(chromosome)
        mutations_made = False
        
        for i in range(length):
            if random.random() < mutation_rate:
                mutations_made = True
                j = random.randint(0, length - 1)
                if i != j:
                    gene_i = Gen(
                        group=new_chromosome[i].group,
                        lesson=new_chromosome[i].lesson,
                        room=new_chromosome[j].room,
                        slot=new_chromosome[j].slot
                    )
                    
                    gene_j = Gen(
                        group=new_chromosome[j].group,
                        lesson=new_chromosome[j].lesson,
                        room=new_chromosome[i].room,
                        slot=new_chromosome[i].slot
                    )
                    
                    new_chromosome[i] = gene_i
                    new_chromosome[j] = gene_j
        
        if not mutations_made or not validate or is_valid_chromosome(new_chromosome):
            return new_chromosome
            
        attempts += 1
    
    print(f"Warning: Could not find valid mutation after {max_attempts} attempts")
    return original_chromosome


def group_reverse_mutation(
    chromosome: Chromosome,
    mutation_rate: float = 0.1,
    validate: bool = True,
    max_attempts: int = 100
) -> Chromosome:
    original_chromosome = chromosome[:]
    attempts = 0
    
    # Get unique groups
    groups = list(set(gene.group for gene in chromosome))
    
    while attempts < max_attempts:
        new_chromosome = chromosome[:]
        mutations_made = False
        
        for group in groups:
            if random.random() < mutation_rate:
                mutations_made = True
                
                # Get all genes for this group
                group_indices = [
                    i for i, gene in enumerate(new_chromosome) 
                    if gene.group == group
                ]
                
                # Get slots and rooms in reverse order
                group_genes = [new_chromosome[i] for i in group_indices]
                reversed_slots = [gene.slot for gene in group_genes][::-1]
                reversed_rooms = [gene.room for gene in group_genes][::-1]
                
                # Create new genes with reversed slots and rooms
                for idx, original_idx in enumerate(group_indices):
                    new_chromosome[original_idx] = Gen(
                        group=group_genes[idx].group,
                        lesson=group_genes[idx].lesson,
                        room=reversed_rooms[idx],
                        slot=reversed_slots[idx]
                    )
        
        if not mutations_made or not validate or is_valid_chromosome(new_chromosome):
            return new_chromosome
            
        attempts += 1
    
    print(f"Warning: Could not find valid mutation after {max_attempts} attempts")
    return original_chromosome

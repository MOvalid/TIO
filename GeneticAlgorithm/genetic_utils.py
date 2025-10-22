from math import inf
import random
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from models import Lesson, Slot, Gen, Chromosome
from collections import defaultdict

def generate_random_chromosome(groups: List[str], lessons: List[Lesson],
                               rooms: List[str], slots: List[Slot]) -> Chromosome:
    chromosome: Chromosome = []

    for group in groups:
        for lesson in lessons:
            room = random.choice(rooms)
            slot = random.choice(slots)
            gene = Gen(group=group, lesson=lesson, room=room, slot=slot)
            chromosome.append(gene)

    return chromosome

def calculate_concentration_bonus(chromosome: Chromosome, max_bonus: int = 100) -> int:
    bonus = 0

    group_day_slots = defaultdict(list)
    teacher_day_slots = defaultdict(list)

    for gene in chromosome:
        group_day_slots[(gene.group, gene.slot.day)].append(gene.slot.start_time)
        teacher_day_slots[(gene.lesson.teacher, gene.slot.day)].append(gene.slot.start_time)

    def entity_bonus(day_slots):
        entity_bonus_sum = 0
        for slots in day_slots.values():
            sorted_slots = sorted(slots)
            if not sorted_slots:
                continue

            total_possible_gaps = len(sorted_slots) - 1
            actual_gaps = 0
            for i in range(1, len(sorted_slots)):
                prev_hour = int(sorted_slots[i-1][:2])
                curr_hour = int(sorted_slots[i][:2])
                gap = curr_hour - prev_hour - 1
                if gap > 0:
                    actual_gaps += gap

            if total_possible_gaps > 0:
                concentration = (total_possible_gaps - actual_gaps) / total_possible_gaps
                entity_bonus_sum += int(max_bonus * concentration)
            else:
                entity_bonus_sum += max_bonus
        return entity_bonus_sum

    bonus += entity_bonus(group_day_slots)
    bonus += entity_bonus(teacher_day_slots)

    return bonus


def calculate_fitness(
    chromosome: Chromosome,
    init_score: int = 1000,
    group_conflict_penalty: int = inf,
    teacher_conflict_penalty: int = inf,
    room_conflict_penalty: int = inf,
    group_gap_penalty_short: int = 20,
    group_gap_penalty_long: int = 40,
    teacher_gap_penalty_short: int = 5,
    teacher_gap_penalty_long: int = 10,
    concentration_bonus: int = 100
) -> int:
    score = init_score

    day_time_map: Dict[Tuple[str, str], List[Gen]] = defaultdict(list)
    for gene in chromosome:
        key = (gene.slot.day, gene.slot.start_time)
        day_time_map[key].append(gene)

    for key, genes in day_time_map.items():
        groups_seen = set()
        teachers_seen = set()
        rooms_seen = set()
        for gene in genes:
            if gene.group in groups_seen:
                score -= group_conflict_penalty
            else:
                groups_seen.add(gene.group)

            if gene.lesson.teacher in teachers_seen:
                score -= teacher_conflict_penalty
            else:
                teachers_seen.add(gene.lesson.teacher)

            if gene.room in rooms_seen:
                score -= room_conflict_penalty
            else:
                rooms_seen.add(gene.room)

    score -= check_group_gaps(chromosome, group_gap_penalty_short, group_gap_penalty_long)
    score -= check_teacher_gaps(chromosome, teacher_gap_penalty_short, teacher_gap_penalty_long)

    concentration_bonus = calculate_concentration_bonus(chromosome, max_bonus=concentration_bonus)
    score += concentration_bonus

    return score


def check_teacher_gaps(chromosome: Chromosome, penalty_short: int, penalty_long: int) -> int:
    score = 0
    teacher_day_slots: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for gene in chromosome:
        teacher_day_slots[(gene.lesson.teacher, gene.slot.day)].append(gene.slot.start_time)

    for (teacher, day), slots in teacher_day_slots.items():
        sorted_slots = sorted(slots)
        for i in range(1, len(sorted_slots)):
            prev_hour = int(sorted_slots[i-1][:2])
            curr_hour = int(sorted_slots[i][:2])
            gap = curr_hour - prev_hour - 1
            if gap == 1:
                score += penalty_short
            elif gap > 1:
                score += penalty_long * gap

    return score


def check_group_gaps(chromosome: Chromosome, penalty_short: int, penalty_long: int) -> int:
    score = 0
    group_day_slots: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for gene in chromosome:
        group_day_slots[(gene.group, gene.slot.day)].append(gene.slot.start_time)

    for (group, day), slots in group_day_slots.items():
        sorted_slots = sorted(slots)
        for i in range(1, len(sorted_slots)):
            prev_hour = int(sorted_slots[i-1][:2])
            curr_hour = int(sorted_slots[i][:2])
            gap = curr_hour - prev_hour - 1

            if gap == 1:
                score += penalty_short
            elif gap > 1:
                score += penalty_long * gap

    return score

def generate_valid_chromosome(
    groups: List[str],
    lessons: List[Lesson],
    rooms: List[str],
    slots: List[Slot],
    max_attempts: int = 1000
) -> Optional[Chromosome]:
    chromosome: Chromosome = []
    
    # Track used slots for each entity
    group_slots: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)    # group -> (day, time)
    teacher_slots: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)  # teacher -> (day, time)
    room_slots: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)     # room -> (day, time)
    
    for group in groups:
        for lesson in lessons:
            # Try to find valid slot and room
            placed = False
            attempts = 0
            
            while not placed and attempts < max_attempts:
                attempts += 1
                
                # Choose random slot and room
                slot = random.choice(slots)
                room = random.choice(rooms)
                time_key = (slot.day, slot.start_time)
                
                # Check if slot is available for group, teacher and room
                if (time_key not in group_slots[group] and 
                    time_key not in teacher_slots[lesson.teacher] and 
                    time_key not in room_slots[room]):
                    
                    # Create and add new gene
                    gene = Gen(group=group, lesson=lesson, room=room, slot=slot)
                    chromosome.append(gene)
                    
                    # Mark slot as used
                    group_slots[group].add(time_key)
                    teacher_slots[lesson.teacher].add(time_key)
                    room_slots[room].add(time_key)
                    
                    placed = True
            
            if not placed:
                # Could not place lesson after max attempts
                return None
    
    return chromosome

def is_valid_chromosome(chromosome: Chromosome) -> bool:
    day_time_map: Dict[Tuple[str, str], List[Gen]] = defaultdict(list)
    
    for gene in chromosome:
        key = (gene.slot.day, gene.slot.start_time)
        day_time_map[key].append(gene)
    
    for genes in day_time_map.values():
        groups_seen = set()
        teachers_seen = set()
        rooms_seen = set()
        
        for gene in genes:
            if (gene.group in groups_seen or
                gene.lesson.teacher in teachers_seen or
                gene.room in rooms_seen):
                return False
            
            groups_seen.add(gene.group)
            teachers_seen.add(gene.lesson.teacher)
            rooms_seen.add(gene.room)
    
    return True

def compare_chromosomes(chromosome1, chromosome2, name1="Original", name2="Modified"):
    """
    Compares two chromosomes and prints detailed differences
    """
    print(f"\nComparing {name1} vs {name2}:")
    print("-" * 50)
    
    differences = 0
    for idx, (gene1, gene2) in enumerate(zip(chromosome1, chromosome2)):
        if gene1 != gene2:
            differences += 1
            print(f"\nGene {idx} differs:")
            print(f"{name1}: {gene1}")
            print(f"{name2}: {gene2}")
            
            # Detail what exactly is different
            if gene1.group != gene2.group:
                print(f"Different group: {gene1.group} -> {gene2.group}")
            if gene1.lesson != gene2.lesson:
                print(f"Different lesson: {gene1.lesson.name} -> {gene2.lesson.name}")
            if gene1.room != gene2.room:
                print(f"Different room: {gene1.room} -> {gene2.room}")
            if gene1.slot != gene2.slot:
                print(f"Different slot: {gene1.slot.day} {gene1.slot.start_time} -> {gene2.slot.day} {gene2.slot.start_time}")
    
    percentage = (differences/len(chromosome1))*100
    print("\nSummary:")
    print(f"Total differences: {differences}/{len(chromosome1)} genes ({percentage:.2f}%)")
    print("-" * 50)

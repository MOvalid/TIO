import random
from typing import List, Dict, Tuple
from collections import defaultdict
from models import Lesson, Slot, Gen, Chromosome

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


def calculate_fitness(
    chromosome: Chromosome,
    init_score: int = 1000,
    group_conflict_penalty: int = 100,
    teacher_conflict_penalty: int = 100,
    room_conflict_penalty: int = 100,
    group_gap_penalty_short: int = 20,
    group_gap_penalty_long: int = 40,
    teacher_gap_penalty_short: int = 5,
    teacher_gap_penalty_long: int = 10
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

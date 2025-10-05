import random
from typing import List
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


def check_group_gaps(chromosome: Chromosome, penalty: int) -> int:
    score = 0
    for group in set(g.group for g in chromosome):
        group_slots = sorted(
            [g.slot.start_time for g in chromosome if g.group == group],
            key=lambda x: x
        )
        for k in range(1, len(group_slots)):
            prev_hour = int(group_slots[k-1][:2])
            curr_hour = int(group_slots[k][:2])
            if curr_hour - prev_hour > 1:
                score += penalty
    return score


def calculate_fitness(
    chromosome: Chromosome,
    init_score: int = 1000,
    group_conflict_penalty: int = 100,
    teacher_conflict_penalty: int = 100,
    room_conflict_penalty: int = 100,
    gap_penalty: int = 10
) -> int:
    score: int = init_score

    for i, g1 in enumerate(chromosome):
        for j, g2 in enumerate(chromosome):
            if i >= j:
                continue
            score -= check_group_conflict(g1, g2, group_conflict_penalty)
            score -= check_teacher_conflict(g1, g2, teacher_conflict_penalty)
            score -= check_room_conflict(g1, g2, room_conflict_penalty)

    score -= check_group_gaps(chromosome, gap_penalty)

    return score


def check_group_conflict(g1: Gen, g2: Gen, penalty: int) -> int:
    if g1.group == g2.group and g1.slot.day == g2.slot.day and g1.slot.start_time == g2.slot.start_time:
        return penalty
    return 0


def check_teacher_conflict(g1: Gen, g2: Gen, penalty: int) -> int:
    if g1.lesson.teacher == g2.lesson.teacher and g1.slot.day == g2.slot.day and g1.slot.start_time == g2.slot.start_time:
        return penalty
    return 0


def check_room_conflict(g1: Gen, g2: Gen, penalty: int) -> int:
    if g1.room == g2.room and g1.slot.day == g2.slot.day and g1.slot.start_time == g2.slot.start_time:
        return penalty
    return 0

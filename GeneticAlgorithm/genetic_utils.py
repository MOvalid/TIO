import random
from typing import List
from models import Lesson, Slot, Gen, Chromosome

INIT_SCORE = 100

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
    gap_penalty: int = 10
) -> int:
    score: int = init_score
    
    # Conflicts
    for i, gene1 in enumerate(chromosome):
        for j, gene2 in enumerate(chromosome):
            if i >= j:
                continue
            if gene1.group == gene2.group and gene1.slot.day == gene2.slot.day and gene1.slot.start_time == gene2.slot.start_time:
                score -= group_conflict_penalty
            if gene1.lesson.teacher == gene2.lesson.teacher and gene1.slot.day == gene2.slot.day and gene1.slot.start_time == gene2.slot.start_time:
                score -= teacher_conflict_penalty
            if gene1.room == gene2.room and gene1.slot.day == gene2.slot.day and gene1.slot.start_time == gene2.slot.start_time:
                score -= room_conflict_penalty

    for group in set(g.group for g in chromosome):
        group_slots = sorted(
            [g.slot.start_time for g in chromosome if g.group == group],
            key=lambda x: x
        )
        for k in range(1, len(group_slots)):
            prev_hour = int(group_slots[k-1][:2])
            curr_hour = int(group_slots[k][:2])
            if curr_hour - prev_hour > 1:
                score -= gap_penalty

    return score

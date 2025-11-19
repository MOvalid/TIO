from typing import List, Dict, Tuple
from collections import defaultdict
from models import Gen, Chromosome
from collections import defaultdict

def calculate_concentration_bonus(chromosome: Chromosome, max_bonus: int = 300) -> int:
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

    # group_conflict_penalty: int = 300,
    # teacher_conflict_penalty: int = 200,
    # room_conflict_penalty: int = 150,

    # group_gap_penalty_short: int = 10,
    # group_gap_penalty_long: int = 20,
    # teacher_gap_penalty_short: int = 5,
    # teacher_gap_penalty_long: int = 10,

    # concentration_bonus: int = 30

    group_conflict_penalty: int = 150,
    teacher_conflict_penalty: int = 120,
    room_conflict_penalty: int = 100,

    group_gap_penalty_short: int = 5,
    group_gap_penalty_long: int = 10,
    teacher_gap_penalty_short: int = 2,
    teacher_gap_penalty_long: int = 5,

    concentration_bonus: int = 30
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

    bonus = calculate_concentration_bonus(chromosome, max_bonus=concentration_bonus)
    score += bonus

    return score


def check_teacher_gaps(chromosome: Chromosome, penalty_short: int, penalty_long: int) -> int:
    score = 0
    teacher_day_slots: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for gene in chromosome:
        teacher_day_slots[(gene.lesson.teacher, gene.slot.day)].append(gene.slot.start_time)

    for (_, _), slots in teacher_day_slots.items():
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

    for (_, _), slots in group_day_slots.items():
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

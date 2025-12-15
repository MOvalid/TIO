from typing import List
from collections import defaultdict
from models import Lesson, Gen, Slot, Chromosome


def gaps_penalty(slots: List[str], penalty_short: int, penalty_long: int) -> int:
    """
    Oblicza karę za przerwy w harmonogramie dla danej listy slotów czasowych.
    """
    if not slots or len(slots) == 1:
        return 0

    slots = sorted(slots)
    score = 0

    for i in range(1, len(slots)):
        prev_hour = int(slots[i - 1][:2])
        curr_hour = int(slots[i][:2])
        gap = curr_hour - prev_hour - 1

        if gap == 1:
            score += penalty_short
        elif gap > 1:
            score += penalty_long * gap

    return score


def concentration_local_bonus(slots: List[str], max_bonus: int) -> int:
    """
    Oblicza lokalną premię za koncentrację (ciągłość) slotów danego podmiotu.
    """
    if len(slots) <= 1:
        return max_bonus

    sorted_slots = sorted(slots)
    total_possible_gaps = len(sorted_slots) - 1
    actual_gaps = 0

    for i in range(1, len(sorted_slots)):
        prev_hour = int(sorted_slots[i - 1][:2])
        curr_hour = int(sorted_slots[i][:2])
        gap = curr_hour - prev_hour - 1
        if gap > 0:
            actual_gaps += gap

    concentration = (total_possible_gaps - actual_gaps) / total_possible_gaps
    return int(max_bonus * concentration)


def heuristic_value(
    lesson: Lesson,
    group: str,
    room: str,
    slot: Slot,
    current_solution: Chromosome,

    group_conflict_penalty: int = 150,
    teacher_conflict_penalty: int = 120,
    room_conflict_penalty: int = 100,

    group_gap_penalty_short: int = 5,
    group_gap_penalty_long: int = 10,
    teacher_gap_penalty_short: int = 2,
    teacher_gap_penalty_long: int = 5,

    concentration_bonus: int = 30
) -> float:
    """
    Heurystyka ACO oceniająca jakość decyzji:
    'Czy umieszczenie lekcji L w sali R o czasie S jest dobrą decyzją?'

    Zwraca wartość dodatnią — im wyżej, tym bardziej atrakcyjna decyzja.
    """
    penalty = 0
    bonus = 0

    for gene in current_solution:
        if gene.slot.day == slot.day and gene.slot.start_time == slot.start_time:

            if gene.group == group:
                penalty += group_conflict_penalty

            if gene.lesson.teacher == lesson.teacher:
                penalty += teacher_conflict_penalty

            if gene.room == room:
                penalty += room_conflict_penalty

    group_slots = [
        g.slot.start_time
        for g in current_solution
        if g.group == group and g.slot.day == slot.day
    ] + [slot.start_time]

    penalty += gaps_penalty(group_slots,
                            group_gap_penalty_short,
                            group_gap_penalty_long)

    teacher_slots = [
        g.slot.start_time
        for g in current_solution
        if g.lesson.teacher == lesson.teacher and g.slot.day == slot.day
    ] + [slot.start_time]

    penalty += gaps_penalty(teacher_slots,
                            teacher_gap_penalty_short,
                            teacher_gap_penalty_long)

    bonus += concentration_local_bonus(group_slots, concentration_bonus)
    bonus += concentration_local_bonus(teacher_slots, concentration_bonus)

    value = bonus - penalty
    return max(1, value)

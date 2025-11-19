from typing import List
from models import Gen

def display_timetable(chromosome: List[Gen]):
    """
    Displays the timetable in a readable, aligned format with vertical bars.
    The output is sorted by day, start time, and group.
    """
    sorted_plan = sorted(chromosome, key=lambda g: (g.slot.day, g.slot.start_time, g.group))

    group_width = max(len("Group"), max(len(gen.group) for gen in sorted_plan)) + 2
    lesson_width = max(len("Lesson"), max(len(gen.lesson.name) for gen in sorted_plan)) + 2
    teacher_width = max(len("Teacher"), max(len(gen.lesson.teacher) for gen in sorted_plan)) + 2
    room_width = max(len("Room"), max(len(gen.room) for gen in sorted_plan)) + 2
    day_width = max(len("Day"), max(len(gen.slot.day) for gen in sorted_plan)) + 2
    time_width = max(len("Time"), max(len(gen.slot.start_time + "-" + gen.slot.end_time) for gen in sorted_plan)) + 2

    header = (f"| {'Group':<{group_width}}"
              f"| {'Lesson':<{lesson_width}}"
              f"| {'Teacher':<{teacher_width}}"
              f"| {'Room':<{room_width}}"
              f"| {'Day':<{day_width}}"
              f"| {'Time':<{time_width}}|")
    separator = "-" * len(header)

    print("\nBest Timetable:\n")
    print(separator)
    print(header)
    print(separator)

    for gen in sorted_plan:
        time_str = f"{gen.slot.start_time}-{gen.slot.end_time}"
        row = (f"| {gen.group:<{group_width}}"
               f"| {gen.lesson.name:<{lesson_width}}"
               f"| {gen.lesson.teacher:<{teacher_width}}"
               f"| {gen.room:<{room_width}}"
               f"| {gen.slot.day:<{day_width}}"
               f"| {time_str:<{time_width}}|")
        print(row)
    print(separator)

from typing import List
from attr import dataclass

@dataclass
class Lesson:
    name: str
    teacher: str

@dataclass
class Slot:
    day: str
    start_time: str
    end_time: str

@dataclass
class Gen:
    group: str
    lesson: Lesson
    room: str
    slot: Slot

Chromosome = List[Gen]
AlgorithmData = tuple[list[str], list[Lesson], list[str], list[Slot]]

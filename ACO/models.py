from typing import Dict, List, TypedDict, Tuple
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


BestParams = Tuple[int, int, float, float, float]
BestStats = List[Dict[str, float]]

@dataclass
class ResultEntry(TypedDict):
    params: BestParams
    best_score: float
    mean_score: float
    iteration_stats: BestStats

AllResults = List[ResultEntry]

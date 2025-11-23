import random
from typing import List, Callable, Dict, Any
from models import Chromosome, Gen, Lesson, Slot

class AntColonyPlanner:
    """
    Ant Colony Optimization planner for generating timetables.
    
    This class constructs timetables using the Ant Colony Optimization (ACO) algorithm.
    It allows using a custom fitness function to evaluate the quality of generated solutions.
    
    Attributes:
        groups: List of student groups.
        lessons: List of lessons.
        rooms: List of available rooms.
        slots: List of available time slots.
        fitness_function: Function used to evaluate a chromosome.
        n_ants: Number of ants used per iteration.
        n_iterations: Number of iterations to run the algorithm.
        evaporation_rate: Rate at which pheromones evaporate each iteration.
        pheromones: Dictionary storing pheromone levels for each lesson-room-slot combination.
        min_pheromone: Lower bound for pheromone levels to prevent stagnation.
        alpha: Influence of pheromone on choice probabilities.
        beta: Influence of heuristic information on choice probabilities.
    """

    def __init__(
        self, 
        groups: List[str], 
        lessons: List[Lesson], 
        rooms: List[str], 
        slots: List[Slot],
        fitness_function: Callable[[Chromosome], int],
        n_ants: int = 10, 
        n_iterations: int = 50, 
        evaporation_rate: float = 0.1,
        alpha = 1.0,
        beta = 2.0
    ):
        """
        Initializes the Ant Colony Planner with a custom fitness function.
        
        Args:
            groups: List of student groups.
            lessons: List of lessons.
            rooms: List of available rooms.
            slots: List of available time slots.
            fitness_function: Function to evaluate the quality of a chromosome.
            n_ants: Number of ants per iteration.
            n_iterations: Total number of iterations.
            evaporation_rate: Pheromone evaporation rate (between 0 and 1).
            alpha: Positive number value of influence of pheromone on choice probabilities.
            beta: Positive number value of influence of heuristic information on choice probabilities.
        """
        self.groups = groups
        self.lessons = lessons
        self.rooms = rooms
        self.slots = slots
        self.fitness_function = fitness_function
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.min_pheromone = 0.00001
        self.pheromones = {
            (lesson.name, room, slot.day, slot.start_time): 1.0
            for lesson in lessons for room in rooms for slot in slots
        }

    def construct_solution(self) -> Chromosome:
        """
        Constructs a timetable (chromosome) for one ant using pheromone-guided probabilistic selection.
        
        Returns:
            Chromosome: A list of Gen objects representing the timetable for all groups and lessons.
        """
        chromosome: Chromosome = []

        for group in self.groups:
            for lesson in self.lessons:
                choices = [(room, slot) for room in self.rooms for slot in self.slots]
                weights = []

                for room, slot in choices:
                    key = (lesson.name, room, slot.day, slot.start_time)
                    pheromone = max(self.pheromones[key], self.min_pheromone)

                    heuristic = 1.0 / (1 + len([g for g in chromosome if g.slot == slot]))

                    weight = (pheromone ** self.alpha) * (heuristic ** self.beta)
                    weights.append(weight)

                chosen_room, chosen_slot = random.choices(choices, weights=weights, k=1)[0]
                chromosome.append(Gen(group=group, lesson=lesson, room=chosen_room, slot=chosen_slot))

        return chromosome


    def update_pheromones(self, best_solution: Chromosome, best_score: int):
        """
        Updates pheromone levels after each iteration based on the best solution found.
        Ensures pheromone levels never fall below the defined minimum threshold.
        
        Args:
            best_solution: The best chromosome found in the current iteration.
            best_score: Fitness score of the best_solution.
        """
        for key in self.pheromones:
            self.pheromones[key] *= (1 - self.evaporation_rate)
            if self.pheromones[key] < self.min_pheromone:
                self.pheromones[key] = self.min_pheromone

        reward = 1.0 / (1 + max(0, -best_score))

        for gen in best_solution:
            key = (gen.lesson.name, gen.room, gen.slot.day, gen.slot.start_time)
            self.pheromones[key] += reward


    def run(self) -> Dict[str, Any]:
        """Executes the Ant Colony Optimization algorithm. Returns a dict with the best solution, its score, and iteration stats.
        
        Returns:
            Dict containing the best solution, its score, and iteration statistics.
        """

        global_best: Chromosome = []
        global_best_score = float("-inf")
        iteration_stats = []

        for iteration in range(self.n_iterations):
            ants_solutions = [self.construct_solution() for _ in range(self.n_ants)]
            scores = [self.fitness_function(sol) for sol in ants_solutions]

            best_index = scores.index(max(scores))
            best_solution = ants_solutions[best_index]
            best_score = scores[best_index]
            mean_score = sum(scores) / len(scores)

            if best_score > global_best_score:
                global_best_score = best_score
                global_best = best_solution

            self.update_pheromones(best_solution, best_score)

            iteration_stats.append({
                "iteration": iteration + 1,
                "best_score": best_score,
                "mean_score": mean_score
            })

        return {
            "best_solution": global_best,
            "best_score": global_best_score,
            "iteration_stats": iteration_stats
        }

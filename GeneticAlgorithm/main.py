


from data import prepare_data, prepare_data_hard, prepare_data_hard2, prepare_data_medium
from genetic_algorithm import genetic_algorithm
from genetic_utils import calculate_fitness, generate_random_chromosome
from models import Chromosome
from utils import display_schedule, display_schedule_table, show_schedule_gui


if __name__ == "__main__":
    groups, lessons, rooms, slots = prepare_data_hard2()
    chromosome: Chromosome = generate_random_chromosome(groups, lessons, rooms, slots)

    fitness_params = {
        "init_score": 1000,
        "group_conflict_penalty": 200,
        "teacher_conflict_penalty": 150,
        "room_conflict_penalty": 100,
        "gap_penalty": 5
    }

    best_chromosome, best_fitness = genetic_algorithm(
        groups=groups,
        lessons=lessons,
        rooms=rooms,
        slots=slots,
        population_size=75,
        generations=1500,
        crossover_rate=0.9,
        mutation_rate=0.1,
        fitness_kwargs=fitness_params
    )

    print("Best fitness:", best_fitness)
    display_schedule(best_chromosome)
    show_schedule_gui(best_chromosome)

from data import prepare_data_medium
from genetic_utils import calculate_fitness, compare_chromosomes, generate_random_chromosome, generate_valid_chromosome
from genetic_operators import group_reverse_mutation, mutate, one_point_crossover, swap_mutation, two_point_crossover, uniform_crossover

groups_list, lessons_list, rooms_list, slots_list = prepare_data_medium()
chromosome1 = generate_valid_chromosome(groups_list, lessons_list, rooms_list, slots_list)
chromosome2 = generate_valid_chromosome(groups_list, lessons_list, rooms_list, slots_list)

print("Chromosome 1:")
print("Fitness:", calculate_fitness(chromosome1))
for idx, gene in enumerate(chromosome1):
    print(f"{idx}. {gene}")

print("\nChromosome 2:")
print("Fitness:", calculate_fitness(chromosome2))
for idx, gene in enumerate(chromosome2):
    print(f"{idx}. {gene}")

# new_chromosome1, new_chromosome2 = one_point_crossover(chromosome1, chromosome2)

# print("\nOne-Point Crossover Result:")
# print("Fitness:", calculate_fitness(new_chromosome1))
# for idx, gene in enumerate(new_chromosome1):
#     print(f"{idx}. {gene}")

# print("\n")
# print("Fitness:", calculate_fitness(new_chromosome2))
# for idx, gene in enumerate(new_chromosome2):
#     print(f"{idx}. {gene}")

# new_chromosome1, new_chromosome2 = two_point_crossover(chromosome1, chromosome2)

# print("\nTwo-Point Crossover Result:")
# print("Fitness:", calculate_fitness(new_chromosome1))
# for idx, gene in enumerate(new_chromosome1):
#     print(f"{idx}. {gene}")

# print("\n")
# print("Fitness:", calculate_fitness(new_chromosome2))
# for idx, gene in enumerate(new_chromosome2):
#     print(f"{idx}. {gene}")

# new_chromosome1, new_chromosome2 = uniform_crossover(chromosome1, chromosome2)

# print("\nUniform Crossover Result:")
# print("Fitness:", calculate_fitness(new_chromosome1))
# for idx, gene in enumerate(new_chromosome1):
#     print(f"{idx}. {gene}")

# print("\n")
# print("Fitness:", calculate_fitness(new_chromosome2))
# for idx, gene in enumerate(new_chromosome2):
#     print(f"{idx}. {gene}")


# print("\nComparing chromosomes:")
# differences1 = sum(1 for g1, g2 in zip(chromosome1, new_chromosome1) if g1 != g2)
# differences2 = sum(1 for g1, g2 in zip(chromosome2, new_chromosome2) if g1 != g2)

# print(f"Changes in chromosome1: {differences1}/{len(chromosome1)} genes ({(differences1/len(chromosome1))*100:.2f}%)")
# print(f"Changes in chromosome2: {differences2}/{len(chromosome2)} genes ({(differences2/len(chromosome2))*100:.2f}%)")

# muted_chromosome1 = mutate(chromosome1, rooms=rooms_list, slots=slots_list, room_mutation_rate=0.1, slot_mutation_rate=0.1)
# muted_chromosome1 = swap_mutation(chromosome1, mutation_rate=0.1)
muted_chromosome1 = group_reverse_mutation(chromosome1, mutation_rate=0.1)
print("\nMuted chromosome 1:")
print("Fitness:", calculate_fitness(muted_chromosome1))
for idx, gene in enumerate(muted_chromosome1 ):
    print(f"{idx}. {gene}")

print("\nComparing original and mutated chromosome:")
compare_chromosomes(chromosome1, muted_chromosome1, "Original", "Mutated")

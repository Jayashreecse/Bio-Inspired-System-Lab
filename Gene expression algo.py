import random

# Parameters
POP_SIZE = 100          # Population size
NUM_GENES = 10          # Number of genes (binary length)
NUM_GENERATIONS = 30    # Number of generations (as requested)
CROSSOVER_RATE = 0.7    # Crossover probability
MUTATION_RATE = 0.01    # Mutation probability
TOURNAMENT_SIZE = 5     # Tournament selection size
ELITISM_COUNT = 10      # Number of elite individuals to preserve


# Fitness Function: maximize negative x^2 to minimize x^2
def fitness(individual):
    x = gene_expression(individual)
    return -(x - 5)**2  # Maximize negative squared distance from 5


# Gene Expression: decode binary chromosome to real number in [0, 10]
def gene_expression(individual):
    binary_str = ''.join(str(bit) for bit in individual)
    value = int(binary_str, 2) / (2**NUM_GENES - 1) * 10
    return value


# Initialize population with random binary sequences
def initialize_population():
    return [[random.randint(0, 1) for _ in range(NUM_GENES)] for _ in range(POP_SIZE)]


# Tournament Selection
def tournament_selection(population):
    tournament = random.sample(population, TOURNAMENT_SIZE)
    tournament.sort(key=fitness, reverse=True)
    return tournament[0]


# Single-point Crossover
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, NUM_GENES - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        return parent1, parent2


# Bit-flip Mutation
def mutate(individual):
    return [1 - gene if random.random() < MUTATION_RATE else gene for gene in individual]


# Elitism: keep top ELITISM_COUNT individuals
def elitism_replacement(old_pop, new_pop):
    combined = old_pop + new_pop
    combined.sort(key=fitness, reverse=True)
    return combined[:POP_SIZE]


# Main GEA algorithm
def gea():
    population = initialize_population()
    best_solution = None
    best_fitness = float('-inf')

    for generation in range(NUM_GENERATIONS):
        offspring = []

        for _ in range(POP_SIZE // 2):
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)

            offspring.extend([child1, child2])

        population = elitism_replacement(population, offspring)

        current_best = max(population, key=fitness)
        current_best_fitness = fitness(current_best)

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best

        print(f"Generation {generation+1}: Best Fitness = {best_fitness:.6f}, x = {gene_expression(best_solution):.4f}")

    return best_solution


# Run the algorithm
best = gea()
print("\nBest solution found:")
print("Binary Chromosome:", ''.join(str(bit) for bit in best))
print("Decoded x value:", gene_expression(best))
print("Fitness (negative x^2):", fitness(best))

import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Euclidean Distance between two points (cities)
def euclidean_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# Fitness Function: Total Distance of the Tour (Sum of distances between consecutive cities)
def fitness(tour, cities):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += euclidean_distance(cities[tour[i]], cities[tour[i + 1]])
    total_distance += euclidean_distance(cities[tour[-1]], cities[tour[0]])  # Return to start city
    return total_distance

# Levy Flight for Solution Perturbation (TSP version)
def levy_flight(dim, beta=1.5):
    u = np.random.normal(0, 1, dim)
    v = np.random.normal(0, 1, dim)
    step = u / (np.abs(v) ** (1 / beta))
    return step

# Generate a random initial population of solutions (permutations of city indices)
def initialize_population(n, dim):
    population = []
    for _ in range(n):
        tour = np.random.permutation(dim)
        population.append(tour)
    return population

# Cuckoo Search Algorithm for TSP
def cuckoo_search(cities, n, Pa, Maxt):
    dim = len(cities)  # Number of cities
    nests = initialize_population(n, dim)  # Initialize population of nests (tours)
    fitness_values = np.array([fitness(nest, cities) for nest in nests])
    
    best_solution_idx = np.argmin(fitness_values)
    best_solution = nests[best_solution_idx]
    best_fitness = fitness_values[best_solution_idx]
    
    t = 0
    while t < Maxt:
        t += 1
        
        # Step 4: Generate new solutions (cuckoos) using Levy flight
        new_nests = np.copy(nests)
        for i in range(n):
            step = levy_flight(dim)
            # Perturb the current tour by swapping elements using Levy flight
            swap_idx = np.random.choice(dim, size=2, replace=False)
            new_nests[i][swap_idx] = new_nests[i][swap_idx[::-1]]
        
        # Step 5: Evaluate fitness of the new solutions
        new_fitness_values = np.array([fitness(nest, cities) for nest in new_nests])
        
        # Step 6: If cuckoo's egg is better, replace the host egg
        for i in range(n):
            if new_fitness_values[i] < fitness_values[i]:
                nests[i] = new_nests[i]
                fitness_values[i] = new_fitness_values[i]
        
        # Step 7: Abandon the worst nests with probability Pa
        for i in range(n):
            if np.random.rand() < Pa:
                worst_idx = np.argmax(fitness_values)
                nests[worst_idx] = np.random.permutation(dim)  # Generate a new random tour
                fitness_values[worst_idx] = fitness(nests[worst_idx], cities)
        
        # Step 8: Keep the best solution found so far
        best_solution_idx = np.argmin(fitness_values)
        best_solution = nests[best_solution_idx]
        best_fitness = fitness_values[best_solution_idx]
        
        # Print progress
        print(f"Iteration {t}: Best Distance = {best_fitness}")
    
    return best_solution, best_fitness

# Example usage for TSP
# Define cities (coordinates)
cities = np.array([
    [0, 0], [1, 2], [2, 4], [3, 1], [4, 3], [5, 0], [6, 2], [7, 4]
])

n = 20  # Number of host nests (solutions)
Pa = 0.25  # Probability of discovering cuckoo's egg
Maxt = 30  # Maximum number of iterations

best_solution, best_fitness = cuckoo_search(cities, n, Pa, Maxt)
print(f"Best Solution (Tour): {best_solution}")
print(f"Best Distance (Total Tour Length): {best_fitness}")

# Plot the best solution
plt.figure(figsize=(8, 6))
x = cities[best_solution, 0]
y = cities[best_solution, 1]
plt.plot(np.append(x, x[0]), np.append(y, y[0]), marker='o')
plt.title('Best Tour Found by Cuckoo Search Algorithm')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
for i, txt in enumerate(best_solution):
    plt.annotate(txt, (x[i], y[i]))
plt.show()

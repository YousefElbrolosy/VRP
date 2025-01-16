import random
import numpy as np
import matplotlib.pyplot as plt

def calculate_route_distance(route, distance_matrix):
    """Calculate the total distance of a route."""
    if len(route) == 0:
        return 0
    distance = distance_matrix[0][route[0]]  # From depot to the first customer
    for i in range(len(route) - 1):
        distance += distance_matrix[route[i]][route[i + 1]]
    distance += distance_matrix[route[-1]][0]  # Return to the depot
    return distance

def evaluate_solution(solution, distance_matrix, vehicle_capacity, demands):
    """Evaluate the fitness of a VRP solution."""
    total_distance = 0
    penalty = 0

    for route in solution:
        # Calculate route distance
        route_distance = calculate_route_distance(route, distance_matrix)
        total_distance += route_distance

        # Check capacity constraint
        route_demand = sum(demands[customer] for customer in route)
        if route_demand > vehicle_capacity:
            penalty += (route_demand - vehicle_capacity) * 10  # Add penalty for exceeding capacity

    fitness = 1 / (total_distance + penalty)  # Fitness is inversely proportional to cost
    return fitness

def generate_initial_population(num_customers, num_vehicles, pop_size):
    """Generate an initial population of solutions."""
    population = []
    customers = list(range(1, num_customers + 1))  # Exclude the depot (node 0)
    for _ in range(pop_size):
        random.shuffle(customers)
        solution = [[] for _ in range(num_vehicles)]
        for customer in customers:
            random.choice(solution).append(customer)
        population.append(solution)
    return population

def select_parents(population, fitness, num_parents):
    """Select parents based on fitness using roulette wheel selection."""
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]
    selected_indices = np.random.choice(len(population), size=num_parents, p=probabilities)
    return [population[i] for i in selected_indices]

def crossover(parent1, parent2):
    """Perform crossover between two parents to create offspring."""
    num_vehicles = len(parent1)
    child = [[] for _ in range(num_vehicles)]

    for i in range(num_vehicles):
        if random.random() < 0.5:
            child[i] = parent1[i].copy()
        else:
            child[i] = parent2[i].copy()

    # Ensure all customers are assigned
    all_customers = set(customer for route in parent1 for customer in route)
    child_customers = set(customer for route in child for customer in route)
    missing_customers = list(all_customers - child_customers)
    extra_customers = list(child_customers - all_customers)

    for customer in missing_customers:
        random.choice(child).append(customer)
    for customer in extra_customers:
        for route in child:
            if customer in route:
                route.remove(customer)
                break

    return child

def mutate(solution, mutation_rate):
    """Perform mutation by swapping customers between routes."""
    for route in solution:
        if random.random() < mutation_rate and len(route) > 1:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
    return solution

def genetic_algorithm_vrp(distance_matrix, demands, vehicle_capacity, num_vehicles, pop_size=100, generations=500, mutation_rate=0.02):
    """Solve the VRP using a genetic algorithm."""
    num_customers = len(demands) - 1  # Exclude the depot
    population = generate_initial_population(num_customers, num_vehicles, pop_size)

    for generation in range(generations):
        fitness = [evaluate_solution(sol, distance_matrix, vehicle_capacity, demands) for sol in population]
        next_population = []

        for _ in range(pop_size // 2):
            parents = select_parents(population, fitness, 2)
            child1 = crossover(parents[0], parents[1])
            child2 = crossover(parents[1], parents[0])
            next_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])

        population = next_population

    # Find the best solution
    fitness = [evaluate_solution(sol, distance_matrix, vehicle_capacity, demands) for sol in population]
    best_index = np.argmax(fitness)
    return population[best_index], 1 / fitness[best_index]

# Example Data
distance_matrix = [
        [0.00, 36.84, 5.06, 30.63],
        [36.84, 0.00, 24.55, 63.22],
        [5.06, 24.55, 0.00, 15.50],
        [30.63, 63.22, 15.50, 0.00]
    ]
demands = [0, 0, 0, 0]  # Demands for customers, depot demand is 0
vehicle_capacity = 30
num_vehicles = 2

# Solve VRP
best_solution, best_cost = genetic_algorithm_vrp(distance_matrix, demands, vehicle_capacity, num_vehicles)

# Print Results
print("Best Solution (Routes):", best_solution)
print("Best Cost (Distance):", best_cost)

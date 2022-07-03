import random

import numpy as np


# EVOLUTIONARY OPERATORS

def crossover(par1, par2):
    ind1, ind2 = np.copy(par1), np.copy(par2)
    size = min(len(ind1), len(ind2))
    p1, p2 = [0] * size, [0] * size

    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i
    # Choose crossover points
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Apply crossover between cx points
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = ind1[i]
        temp2 = ind2[i]
        # Swap the matched value
        ind1[i], ind1[p1[temp2]] = temp2, temp1
        ind2[i], ind2[p2[temp1]] = temp1, temp2
        # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2


def mutate(p, indpb):
    individual = np.copy(p)
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            swap_indx = random.randint(0, size - 2)
            if swap_indx >= i:
                swap_indx += 1
            individual[i], individual[swap_indx] = \
                individual[swap_indx], individual[i]

    return individual


# MULTIFACTORIAL EVOLUTIONARY HELPER FUNCTIONS
def find_relative(population, skill_factor, sf, N):
    return population[np.random.choice(np.where(skill_factor[:N] == sf)[0])]


def calculate_scalar_fitness(factorial_cost):
    return 1 / np.min(np.argsort(np.argsort(factorial_cost, axis=0), axis=0) + 1, axis=1)


# OPTIMIZATION RESULT HELPERS
def get_best_individual(population, factorial_cost, scalar_fitness, skill_factor, sf):
    # select individuals from task sf
    idx = np.where(skill_factor == sf)[0]
    subpop = population[idx]
    sub_factorial_cost = factorial_cost[idx]
    sub_scalar_fitness = scalar_fitness[idx]

    # select best individual
    idx = np.argmax(sub_scalar_fitness)
    x = subpop[idx]
    fun = sub_factorial_cost[idx, sf]
    return x, fun

from tqdm import trange

from helpers import *
from operators import *
from surrogate import Surrogate

config = load_config()


def mfea(task, config, callback=None):
    functions = task.functions
    K = len(functions)
    N = config['pop_size'] * K
    D = task.dimension
    T = config['num_iter']
    rmp = config['rmp']

    # initialize
    population = np.zeros((2 * N, D), dtype=int)
    for i in range(2 * N):
        population[i] = np.random.permutation(D)

    skill_factor = np.array([i % K for i in range(2 * N)])
    factorial_cost = np.full([2 * N, K], np.inf)
    scalar_fitness = np.empty([2 * N])

    # evaluate
    for i in range(2 * N):
        sf = skill_factor[i]
        factorial_cost[i, sf] = functions[sf](population[i])
    scalar_fitness = calculate_scalar_fitness(factorial_cost)

    task.surrogate_model = Surrogate(len(task.functions), population, skill_factor, factorial_cost, N // 10)

    # sort
    sort_index = np.argsort(scalar_fitness)[::-1]
    population = population[sort_index]
    skill_factor = skill_factor[sort_index]
    factorial_cost = factorial_cost[sort_index]

    # evolve
    iterator = trange(T)
    for t in iterator:
        # permute current population
        permutation_index = np.random.permutation(N)
        population[:N] = population[:N][permutation_index]
        skill_factor[:N] = skill_factor[:N][permutation_index]
        factorial_cost[:N] = factorial_cost[:N][permutation_index]
        factorial_cost[N:] = np.inf

        # select pair to crossover
        for i in range(0, N, 2):
            p1, p2 = population[i], population[i + 1]
            sf1, sf2 = skill_factor[i], skill_factor[i + 1]

            # crossover
            if sf1 == sf2:
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1, rmp)
                c2 = mutate(c2, rmp)
                skill_factor[N + i] = sf1
                skill_factor[N + i + 1] = sf1
            elif sf1 != sf2 and np.random.rand() < rmp:
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1, rmp)
                c2 = mutate(c2, rmp)
                if np.random.rand() < 0.5:
                    skill_factor[N + i] = sf1
                else:
                    skill_factor[N + i] = sf2
                if np.random.rand() < 0.5:
                    skill_factor[N + i + 1] = sf1
                else:
                    skill_factor[N + i + 1] = sf2
            else:
                p2 = find_relative(population, skill_factor, sf1, N)
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1, rmp)
                c2 = mutate(c2, rmp)
                skill_factor[N + i] = sf1
                skill_factor[N + i + 1] = sf1

            population[N + i, :], population[N + i + 1, :] = c1[:], c2[:]

        # evaluate
        for i in range(N, 2 * N):
            sf = skill_factor[i]
            factorial_cost[i, sf] = functions[sf](population[i])
        scalar_fitness = calculate_scalar_fitness(factorial_cost)

        # sort
        sort_index = np.argsort(scalar_fitness)[::-1]
        population = population[sort_index]
        skill_factor = skill_factor[sort_index]
        factorial_cost = factorial_cost[sort_index]
        scalar_fitness = scalar_fitness[sort_index]

        c1 = population[np.where(skill_factor == 0)][0]
        c2 = population[np.where(skill_factor == 1)][0]

        real_factorial_cost = np.array([functions[skill_factor[i]](population[i], False) for i in range(N // 10)])
        task.surrogate_model.update(population[:N // 10], skill_factor[:N // 10], real_factorial_cost)

        # optimization info
        message = {'algorithm': 'mfea', 'rmp': rmp}
        results = get_optimization_results(t, population, factorial_cost, scalar_fitness, skill_factor, message)
        if callback:
            callback(results)

        desc = 'gen:{} factorial_cost:{} message:{}'.format(t, ' '.join('{:0.6f}'.format(res.fun) for res in results),
                                                            message)
        iterator.set_description(desc)

import numpy as np
from sklearn.linear_model import SGDRegressor


class Surrogate:
    def __init__(self, num_task, pop, skill_factor, factorial_cost, n=10):
        self.edge = {}
        self.model = [SGDRegressor() for _ in range(num_task)]
        self.sum = 0
        self.n = int(n)
        self.update_edge(pop)

        for i in range(num_task):
            x = np.array([self.encode(ind) for ind in pop[np.where(skill_factor == i)]])
            y = np.array([val[i] for val in factorial_cost[np.where(skill_factor == i)]])
            self.model[i].fit(x, y)

    def update(self, pop, skill_factor, real_factorial_cost):
        for i, model in enumerate(self.model):
            x = np.array([self.encode(ind) for ind in pop[np.where(skill_factor == i)]])
            y = np.array([val for val in real_factorial_cost[np.where(skill_factor == i)]])
            model.partial_fit(x, y)

        self.update_edge(pop[:self.n])

    def update_edge(self, ind_set):
        # self.sum = 0
        # self.edge = {}
        for ind in ind_set:
            for i in range(-1, len(ind) - 1):
                self.edge[ind[i], ind[i + 1]] = self.edge.get((ind[i], ind[i + 1]), 0) + 1
            self.sum += 1

    def predict(self, ind, func):
        x = np.array([self.encode(ind)])
        return self.model[func].predict(x)

    def encode(self, ind):
        return ind
        # tmp = np.zeros(10)
        # for i in range(-1, len(ind) - 1):
        #     tmp[int(self.edge.get((ind[i], ind[i + 1]), 0) / self.sum * 10)] += 1
        #
        # return tmp

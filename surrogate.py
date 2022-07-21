import numpy as np
from sklearn.svm import SVR


class Surrogate:
    def __init__(self, num_task, pop, skill_factor, factorial_cost):
        self.edge = [{} for _ in range(num_task)]
        self.model = [SVR() for _ in range(num_task)]
        self.sum = 0
        self.update_edge(pop[:len(pop)//10], skill_factor[:len(pop)//10])

        for i in range(num_task):
            x = np.array([self.encode(ind, i) for ind in pop[np.where(skill_factor == i)]])
            y = np.array([val[i] for val in factorial_cost[np.where(skill_factor == i)]])
            self.model[i].fit(x, y)

    def update(self, pop, skill_factor, real_factorial_cost):
        for i, model in enumerate(self.model):
            x = np.array([self.encode(ind, i) for ind in pop[np.where(skill_factor == i)]])
            y = np.array([val for val in real_factorial_cost[np.where(skill_factor == i)]])
            model.fit(x, y)

        self.update_edge(pop, skill_factor)

    def update_edge(self, ind_set, skill_factor):
        # self.sum = 0
        # self.edge = {}
        for s, ind in enumerate(ind_set):
            for i in range(-1, len(ind) - 1):
                self.edge[skill_factor[s]][ind[i], ind[i + 1]] = \
                    self.edge[skill_factor[s]].get((ind[i], ind[i + 1]), 0) + 1
            self.sum += 1

    def predict(self, ind, func):
        x = np.array([self.encode(ind, func)])
        return self.model[func].predict(x)

    def encode(self, ind, func):
        # return ind
        tmp = np.zeros(10)
        for i in range(-1, len(ind) - 1):
            tmp[int(self.edge[func].get((ind[i], ind[i + 1]), 0) / self.sum * 10)] += 1

        return tmp

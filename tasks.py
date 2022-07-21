import os

import numpy as np
from scipy.spatial.distance import cdist

DIRNAME = os.path.dirname(__file__)


class TS_TR:
    def __init__(self, use_surrogate=True):
        self.path = os.path.join(DIRNAME, 'data/att48.tsp')
        file = open(self.path)
        lines = file.readlines()
        self.functions = [self.tsp, self.trp]
        self.dimension = int(lines[3].strip().split(":")[1])
        self.coord = np.zeros((self.dimension, 2), dtype=int)
        for i in range(self.dimension):
            self.coord[i] = [float(_) for _ in lines[6 + i].strip().split()[1:]]

        self.distance = cdist(self.coord, self.coord)
        self.surrogate_model = None
        self.use_surrogate = use_surrogate

    def tsp(self, x, use_surrogate=None):
        if use_surrogate is None:
            use_surrogate = self.use_surrogate
        if use_surrogate and self.surrogate_model is not None:
            return self.surrogate_model.predict(x, self.functions.index(self.tsp))

        return sum([self.distance[x[i], x[i + 1]] for i in range(-1, self.dimension - 1)])

    def trp(self, x, use_surrogate=None):
        if use_surrogate is None:
            use_surrogate = self.use_surrogate
        if use_surrogate and self.surrogate_model is not None:
            return self.surrogate_model.predict(x, self.functions.index(self.trp))
        return sum([self.distance[x[j - 1], x[j]] for i in range(1, self.dimension - 1) for j in range(1, i)])

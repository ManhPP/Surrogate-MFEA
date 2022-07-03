import os

import numpy as np
from scipy.spatial.distance import cdist

DIRNAME = os.path.dirname(__file__)


class TS_TR:
    def __init__(self):
        self.path = os.path.join(DIRNAME, 'data/att48.tsp')
        file = open(self.path)
        lines = file.readlines()
        self.functions = [self.tsp, self.trp]
        self.dimension = int(lines[3].strip().split(":")[1])
        self.coord = np.zeros((self.dimension, 2), dtype=int)
        for i in range(self.dimension):
            self.coord[i] = lines[6 + i].strip().split()[1:]

        self.distance = cdist(self.coord, self.coord)

    def tsp(self, x):
        return sum([self.distance[x[i], x[i + 1]] for i in range(-1, self.dimension - 1)])

    def trp(self, x):
        return sum([self.distance[x[j - 1], x[j]] for i in range(1, self.dimension - 1) for j in range(1, i)])

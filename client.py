from math import inf

import numpy

from server import readFile


def getKMatrix(nodes: list[list[float]]) -> list[list[float]]:
    kmatrix = []
    for i, i_val in enumerate(nodes):
        kmatrix.append([0] * len(nodes))
        x_i, y_i = i_val
        for j, j_val in enumerate(nodes):
            x_j, y_j = j_val
            kmatrix[i][j] = ((x_i - x_j) ** 2 + (y_i - y_j) ** 2) ** 0.5
    return kmatrix


def getRandomNode(probabilities: list[float]):
    return numpy.random.choice(numpy.arange(0, len(probabilities)), p=probabilities)


class AntColony:
    """
    Based on https://www.researchgate.net/publication/277284831_MAX-MIN_ant_system
    """

    TMIN = 0
    TMAX = 1
    PBEST = 0.05
    PERSISTENCE = 0.98
    ALPHA = 1
    BETA = 2

    def __init__(self, kmatrix) -> None:
        kmatrix = getKMatrix(readFile("data/small"))
        self.kmatrix = kmatrix
        self.pheromones = [[self.TMAX for _ in kmatrix] for _ in kmatrix]
        self.global_solution = []
        self.global_cost = inf
        self.solution = []
        self.cost = inf
        self.node_count = len(kmatrix)
        self.ant_count = self.node_count

    def start(self):
        for _ in range(100):
            self.iterate()

    def iterate(self):
        for _ in range(self.ant_count):
            self.walkAnt()
        self.updatePheromones()
        self.cost = inf
        self.solution = []
        print(self.global_solution)
        print(self.global_cost)

    def walkAnt(self):
        current = numpy.random.randint(0, self.node_count)
        solution = [current]
        visited = set(solution)
        cost = 0

        while len(visited) < self.node_count:
            probs = [x if x not in visited else 0 for x in self.getP(current)]
            nextNode = getRandomNode(probs)
            solution.append(nextNode)
            visited.add(nextNode)
            cost += self.kmatrix[current][nextNode]
            current = nextNode

        if cost < self.cost:
            self.cost = cost
            self.solution = solution

        if cost < self.global_cost:
            self.global_cost = cost
            self.global_solution = solution
            self.updateT()

    def getP(self, src):
        probabilities = []
        for i in range(len(self.kmatrix)):
            if src == i:
                probabilities.append(0)
                continue
            pheromone = self.pheromones[src][i]
            distance = self.kmatrix[src][i]
            probabilities.append(
                pheromone ** self.ALPHA * distance ** -self.BETA)
        total = sum(probabilities)
        return [x / total for x in probabilities]

    def updateT(self):
        self.TMAX = 1 / ((1 - self.PERSISTENCE) * self.global_cost)
        pdec = self.PBEST ** (1 / self.ant_count)
        self.TMIN = self.TMAX / (self.ant_count / 2) * (1 - pdec) / pdec

    def updatePheromones(self):
        for i in range(len(self.pheromones)):
            for j in range(len(self.pheromones)):
                self.pheromones[i][j] *= self.PERSISTENCE

        for i in range(len(self.solution) - 1):
            src = self.solution[i]
            dest = self.solution[i + 1]
            self.pheromones[src][dest] += 1 / self.cost
            self.pheromones[dest][src] += 1 / self.cost


if __name__ == "__main__":
    a = AntColony([])
    a.start()

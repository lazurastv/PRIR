from math import inf
from queue import Queue

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
        self.kmatrix = kmatrix
        self.pheromones = [[self.TMAX for _ in kmatrix] for _ in kmatrix]
        self.global_solution = []
        self.global_cost = inf
        self.solution = []
        self.cost = inf
        self.node_count = len(kmatrix)
        self.ant_count = self.node_count
        self.queued_ants = Queue()
        self.iteration = 0

    def start(self):
        for i in range(2500):
            self.iterate()
            if i % 100 == 0:
                print(f"Iteration {i} of 2500: Best cost = {self.global_cost}")

    def iterate(self):
        self.cost = inf
        self.solution = []
        for _ in range(self.ant_count):
            self.walkAnt()
        self.updatePheromones()
        self.iteration += 1

    def walkAnt(self):
        current = numpy.random.randint(0, self.node_count)
        solution = [current]
        visited = set(solution)
        cost = 0

        while len(visited) < self.node_count:
            probs = self.getP(current, visited)
            nextNode = getRandomNode(probs)
            solution.append(nextNode)
            visited.add(nextNode)
            cost += self.kmatrix[current][nextNode]
            current = nextNode

        if cost < self.cost:
            self.cost = cost
            self.solution = solution

        self.updateGlobal(solution, cost)

    def getP(self, src, visited):
        probabilities = []
        for i in range(len(self.kmatrix)):
            if src == i or i in visited:
                probabilities.append(0)
                continue
            pheromone = self.pheromones[src][i]
            distance = self.kmatrix[src][i]
            probabilities.append(
                pheromone ** self.ALPHA * distance ** -self.BETA)
        total = sum(probabilities)
        return [x / total for x in probabilities]

    def updateGlobal(self, solution, cost):
        if cost < self.global_cost:
            self.global_cost = cost
            self.global_solution = solution
            self.updateT()

    def updateT(self):
        self.TMAX = 1 / ((1 - self.PERSISTENCE) * self.global_cost)
        pdec = self.PBEST ** (1 / self.ant_count)
        self.TMIN = self.TMAX / (self.ant_count / 2) * (1 - pdec) / pdec
        if self.TMIN > self.TMAX:
            self.TMIN = self.TMAX

    def layPheromones(self, solution, cost):
        for i in range(len(solution) - 1):
            src = solution[i]
            dest = solution[i + 1]
            self.pheromones[src][dest] += 1 / cost
            self.pheromones[dest][src] += 1 / cost

    def updatePheromones(self):
        for i in range(len(self.pheromones)):
            for j in range(len(self.pheromones)):
                self.pheromones[i][j] *= self.PERSISTENCE

        self.layPheromones(self.solution, self.cost)

        if not self.queued_ants.empty():
            solution, cost = self.queued_ants.get()
            self.layPheromones(solution, cost)
            self.updateGlobal(solution, cost)

        for i in range(len(self.pheromones)):
            for j in range(len(self.pheromones)):
                value = self.pheromones[i][j]
                self.pheromones[i][j] = max(min(value, self.TMAX), self.TMIN)


if __name__ == "__main__":
    # We will receive the data from server, this is temporary
    a = AntColony(getKMatrix(readFile("data/small")))
    a.start()
    print(f"Final cost = {a.global_cost}")
    print(f"Final solution: {a.global_solution}")

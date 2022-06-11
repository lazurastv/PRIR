from time import time_ns
from mpi4py import MPI

from client import AntColony, getKMatrix
from server import readFile

start_time = time_ns()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
cartesian = comm.Create_cart(dims=[size], periods=[True])

left, right = cartesian.Shift(direction=0, disp=1)

if rank == 0:
    data = readFile("data/medium")
    file = open(f"data/results_{size}", "w")
else:
    data = None

data = comm.bcast(data, 0)
kmatrix = getKMatrix(data)
colony = AntColony(kmatrix)
ant = None
for i in range(500):
    colony.iterate()
    solution = colony.solution
    cost = colony.cost
    ant = comm.sendrecv((colony.solution, colony.cost),
                        right, 0, None, left, 0)
    colony.queued_ants.put(ant)
    if rank == 0:
        file.write(f"{i} {colony.global_cost} {time_ns() - start_time}\n")

if rank == 0:
    file.write(f"Best cost: {colony.global_cost}\n")
    file.write("Path taken: ")
    file.write(' '.join([str(x) for x in colony.global_solution]))

from time import time_ns
from mpi4py import MPI

from client import AntColony, getKMatrix
from server import readFile
from sys import argv, stdout

start_time = time_ns()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
cartesian = comm.Create_cart(dims=[size], periods=[True])

left, right = cartesian.Shift(direction=0, disp=1)
if len(argv) < 3:
    if rank == 0:
        print(
            "Usage: clientParallel.py [file name in data/] [parallelization type]")
    exit()
problem = argv[1]
parallel_type = argv[2] if len(argv) > 1 else "ant"

allowed_type = ["colony", "ant"]
if parallel_type not in allowed_type:
    if rank == 0:
        print(
            f"Invalid paralellization type. Choose either {', '.join(allowed_type)}.")
        print("Colony - processes run seperate colonies.")
        print("Ant - processes run seperate ants on one colony.")
    exit()

failed_read = False
if rank == 0:
    try:
        data = readFile("data/" + problem)
    except FileNotFoundError:
        print(f"File data/{problem} doesn't exist.")
        data = None
    if data:
        file = open(f"results/results_{problem}_{parallel_type}_{size}", "w")
        file.write(f"Problem size: {problem}\n")
else:
    data = None

data = comm.bcast(data, 0)
if not data:
    exit()
if len(data) < 2:
    if rank == 0:
        print("The problem must have at least 4 nodes!")
    exit()

kmatrix = getKMatrix(data)
colony = AntColony(kmatrix)
ant = None
if parallel_type == "ant":
    colony.ant_count //= size
    colony.ant_count = max(colony.ant_count, 1)

for i in range(500):
    colony.iterate()
    solution = colony.solution
    cost = colony.cost
    if parallel_type == "ant":
        ant = comm.gather((colony.solution, colony.cost), 0)
        if rank == 0:
            ant.sort(key=lambda x: x[1])
            colony.queued_ants.put(ant[0])
            colony.updatePheromones()
        pheromones = comm.bcast(colony.pheromones, 0)
        colony.pheromones = pheromones
    else:
        ant = comm.sendrecv((colony.solution, colony.cost),
                            right, 0, None, left, 0)
        colony.queued_ants.put(ant)
        colony.updatePheromones()
    if rank == 0:
        file.write(f"{i} {colony.global_cost} {time_ns() - start_time}\n")
        if i % 50 == 0:
            print(f"{i / 5}% complete...")
        stdout.flush()


if rank == 0:
    print(colony.TMAX)
    print(colony.TMIN)
    print("100% complete...")
    print(f"Best cost: {colony.global_cost}")
    file.write(f"Best cost: {colony.global_cost}\n")
    file.write("Path taken: ")
    file.write(' '.join([str(x) for x in colony.global_solution]))

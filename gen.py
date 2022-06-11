from random import randint


COUNT = 50
MAX_COORD = 100
FILENAME = "large"

with open("data/" + FILENAME, "w") as file:
    for i in range(COUNT):
        file.write(f"{randint(0, MAX_COORD)} {randint(0, MAX_COORD)}\n")
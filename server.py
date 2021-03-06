from typing import List

def readFile(name: str) -> List[List[float]]:
    with open(name, 'r') as file:
        text = file.read()
        values = text.split()
        nodes = []
        for i in range(0, len(values), 2):
            nodes.append([float(x) for x in values[i:i+2]])
        return nodes

def getKMatrix(nodes: list[list[float]]) -> list[list[float]]:
    kmatrix = []
    for i, i_val in enumerate(nodes):
        kmatrix.append([0] * len(nodes))
        x_i, y_i = i_val
        for j, j_val in enumerate(nodes):
            x_j, y_j = j_val
            kmatrix[i][j] = ((x_i - x_j) ** 2 + (y_i - y_j) ** 2) ** 0.5
    return kmatrix

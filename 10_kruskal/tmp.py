input = [
    [0, 1, 5],
    [0, 5, 5],
    [0, 4, 4],
    [0, 6, 5],
    [1, 5, 10],
    [1, 2, 7],
    [2, 5, 2],
    [2, 3, 2],
    [3, 5, 3],
    [3, 4, 1],
    [3, 6, 3],
    [4, 5, 2],
    [4, 6, 3],
    [7, 8, 3],
    [7, 13, 8],
    [7, 11, 4],
    [8, 9, 6],
    [8, 10,3],
    [9, 10, 4],
    [9, 12, 5],
    [10, 11, 4],
    [11, 12, 4],
    [12, 13, 3]
]

matrix = [[0 for i in range(14)] for _ in range(14)]

for [u, v, w] in input:
    matrix[u][v] = w

for i in range(len(matrix)):
    for j in range(len(matrix)):
        print(f'{matrix[i][j]}', end=' ')
    print()
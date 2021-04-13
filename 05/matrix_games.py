import numpy as np
import fm

# TODO: Refactor with list indexing
def reconstruct_vector(X, original_length, anulled_indices):
    new_vec = np.zeros(original_length)
    idx = 0
    for i in range(original_length):
        if i not in anulled_indices:
            new_vec[i] = X[idx]
            idx += 1
        
    return new_vec

def get_partial_matrix(matrix, removed_rows, removed_columns):
    m, n = matrix.shape
    # print(f'Asking for partial matrix of dims {m, n} with rr={removed_rows}, rc={removed_columns}')

    include_rows =      [i for i in range(m) if i not in removed_rows]
    include_columns =   [j for j in range(n) if j not in removed_columns]
    return matrix[include_rows, :][:, include_columns]

# Returns a new reduced game matrix
# and indices of dominated rows and columns
def dominate(game_matrix):
    game_matrix = np.array(game_matrix)
    m, n = game_matrix.shape
    removed_rows = []
    removed_columns = []
    while True:
        # Dominate rows
    
        changed = False
        for i in range(m):
            stopi = False
            if i in removed_rows:
                continue
            for j in range(i+1, m):
                if stopi:
                    break
                if j in removed_rows:
                    continue
                rowi = game_matrix[i, [e for e in range(n) if e not in removed_columns]]
                rowj = game_matrix[j, [e for e in range(n) if e not in removed_columns]]
                if (rowi >= rowj).all():
                    # print(f'\trow {rowi} dominated {rowj}')
                    # Remove j-th row
                    removed_rows.append(j)
                    changed = True
                elif (rowj >= rowi).all():
                    # print(f'\trow {rowj} dominated {rowi}')
                    # Remove i-th row
                    removed_rows.append(i)
                    changed = True
                    stopi = True

        # Dominate columns
        for r in range(n):
            stopr = False
            if r in removed_columns:
                continue
            for s in range(r+1, n):
                if stopr:
                    break
                if s in removed_columns:
                    continue
                colr = game_matrix[[e for e in range(m) if e not in removed_rows], r]
                cols = game_matrix[[e for e in range(m) if e not in removed_rows], s]
                if (cols <= colr).all():
                    # print(f'\tcol {cols} dominated {colr}')
                    # Remove r-th column
                    removed_columns.append(r)
                    changed = True
                    stopr = True
                elif (colr <= cols).all():
                    # Remove s-th column
                    # print(f'\tcol {colr} dominated {cols}')
                    removed_columns.append(s)
                    changed = True
        
        # print(get_partial_matrix(game_matrix, removed_rows, removed_columns))
        if not changed:
            break
    
    # game_matrix = game_matrix[[i for i in range(m) if i not in removed_rows], [j for j in range(n) if j not in removed_columns]]
    return get_partial_matrix(game_matrix, removed_rows, removed_columns), removed_rows, removed_columns
    # return game_matrix, removed_rows, removed_columns



def solve_game(game_matrix):
    game_matrix = np.array(game_matrix)
    m, n = game_matrix.shape

    print(f'Solving game for:')
    print(game_matrix)
    game_matrix_dominated, removed_rows, removed_columns = dominate(game_matrix)
    print(f'Dominated:')
    print(game_matrix_dominated)
    


def example1():
    M = [[2, 1, 2, 3],
        [3, 1.5, 1, 2],
        [2, 2, 1, 1],
        [1, 1, 1, 1]]

    solve_game(M)

if __name__ == '__main__':
    print(f'Running...')
    example1()
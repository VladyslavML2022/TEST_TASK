import numpy as np
import argparse

# function for generating random map with specific m and n
def generate_map(m, n):
    return np.random.randint(2, size=(m,n))
    
# function for counting islands
def count_islands(matrix):
    num_islands = 0
    
    # determine function which recursively checks neighbors and marks values as 0 where we've already been 
    def check_neighbors(i, j):
    
        if i < 0 or i >= matrix.shape[0] or j < 0 or j >= matrix.shape[1] or matrix[i,j] == 0:
            return None
        matrix[i,j] = 0
        check_neighbors(i + 1, j)
        check_neighbors(i - 1, j)
        check_neighbors(i, j + 1)
        check_neighbors(i, j - 1)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j] == 1:
                num_islands+=1
                check_neighbors(i,j)

    return num_islands


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Test Count islands algorithm')
    parser.add_argument('--m', type=int, default=3, help='Rows in matrix')
    parser.add_argument('--n', type=int, default=3, help='Columns in matrix')
    args = parser.parse_args()

    matrix = generate_map(args.m, args.n)
    print(f'Generated matrix\n{matrix}')

    # create matrix copy to pass is in the function
    matrix_copy = np.copy(matrix)

    print(f'Number of islands in matrix: {count_islands(matrix_copy)}')
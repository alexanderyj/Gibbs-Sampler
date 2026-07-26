import numpy as np
import scipy.linalg as sp

#Future note: scipy.linalg.solve_triangular does these 2
#Forward substitution to solve Lx=b for lower triangular matrix L
def fsub(L, b):
    x = np.copy(b)
    for i in range(x.size):
        for j in range(i):
            x[i] -= x[j]*L[i][j]
        x[i] /= L[i][i]
    return x

#Backwards substitution to solve Ux=b for uppter triangular matrix U
def bsub(U, b):
    x = np.copy(b)
    n = b.size
    for i in range(n-1,-1,-1):
        for j in range(n-i-1):
            x[i] -= x[i+j+1]*U[i][i+j+1]
        x[i] /= U[i][i]
    return x

U = np.array([[2., 5., 8., 1],
               [0, 3., 4., 2],
               [0, 0, 5, -3],
               [0, 0, 0, -7]])
#print(bsub(U, b))

L = np.array([[2., 0, 0, 0],
               [1, 3., 0, 0],
               [-4, -2, 5, 0],
               [3, 8, 3, -7]])
#print(fsub(L, b))
import numpy as np
import scipy.linalg as sp

#Iterative solver of Ax=b given matrix splitting A = M - N, initial x value of x0, and error bound e
def iterative_solver(M, N, b, x0=None, e=0.01):
    if x0==None:
        x0 = np.empty(b.size)
    r = b - (M-N) @ x0
    x = np.copy(x0)
    while(sp.norm(r, 2) > e):
        #print(r)
        x = x + sp.solve(M, r)
        r = b - (M-N) @ x
    return x

    


#Gauss-Seidel iterative solver of equation Ax=b, initial x value of x0, error bound e
def gauss_seidel(A, b, x0=None, e=0.01):
    #A = L + D + U, M = L + D, N = -U
    bnorm = sp.norm(b, 2)
    M = np.tril(A)
    if x0==None:
        x0 = np.empty(b.size)
    r = b - (A @ x0) #r_0
    x = np.copy(x0) #Initially x_0
    while(sp.norm(r, 2) > e*bnorm):
        #print(r)
        x = x + sp.solve_triangular(M, r, lower=True) #x_(k+1) = x_k + M^-1 r_k
        r = b - (A @ x) #r_(k+1) = b - A * (x_(k+1))
    return x

#Gibbs sampler for multivariate normal distribution with precision matrix A, initial state y0, k_max iterations
#Uses Gauss-Seidel splitting (noise vector has variance D)
def gibbs_sampler(A, mean=None, y0=None, k_max=300):
    #A is assumed to be symmetric positive definite
    n = np.shape(A)[0]
    if y0==None:
        y0 = np.empty(n)
    if mean.all()==None:
        mean = np.zeros(n)
    S = np.sqrt(np.diag(A))
    M = np.tril(A)
    N = -1*np.triu(A, 1)
    y = np.copy(y0)

    c_mean = A @ mean
    for k in range(k_max):
        #Noise vector c_k
        c = np.empty(n)
        for x in range(n):
            c[x] = np.random.normal(c_mean[x], S[x])
        y = sp.solve_triangular(M, (N @ y) + c, lower=True)   
    return y


b = np.array([1.0, -1.0, 0.0, 2.])

A = np.array([[7., 3, 2, 5],
               [3, 4, 1, 3],
               [2, 1, 3, 2],
               [5, 3, 2, 6]])
target_mean = np.array([3, -1, -2, 6])

test_size = 300
test_set = np.empty((test_size, 4))
for x in range(test_size):
    test_set[x] = gibbs_sampler(A, mean=target_mean)

print(np.mean(test_set, axis=0))
cov = np.cov(np.transpose(test_set))
precision = sp.inv(cov)
print(precision)
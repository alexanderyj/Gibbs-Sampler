import numpy as np
import scipy.linalg as sp
import scipy.sparse.linalg as sl
from scipy.sparse.linalg import LinearOperator
import math

#Iterative solver of Ax=b given matrix splitting A = M - N, initial x value of x0, and error bound e
def iterative_solver(M, N, b, x0=None, e=0.01):
    if x0 is None:
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
    if x0 is None:
        x0 = np.zeros(b.size)
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
    if y0 is None:
        y0 = np.empty(n)
    if mean is None:
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

def SSOR_sampler(A, mean=None, y0=None, w=1, k_max=300):
    n = np.shape(A)[0]
    D = np.diag(np.diag(A))
    sqrtD = np.sqrt(D)
    M_SOR = (1/w)*D + np.tril(A, -1)
    N_SOR = ((1-w)/w)*D - np.triu(A, 1)
    c = math.sqrt((2/w - 1))
    z = np.empty(n)
    if mean is None:
        mean = np.zeros(n)
    if y0 is None:
        y = np.zeros(n)
    else:
        y = np.copy(y0)

    for k in range(k_max):
        for a in range(n):
            z[a] = np.random.normal(0, 1)
        x = sp.solve_triangular(M_SOR, N_SOR@y + c*sqrtD@z, lower=True)
        for a in range(n):
            z[a] = np.random.normal(0, 1)
        y = sp.solve_triangular(np.transpose(M_SOR), np.transpose(N_SOR)@x + c*sqrtD@z, lower=False)
    return y+mean

def cheby_acc_SSOR_sampler(A, d_max=None, d_min=None, w=1, mean=None, y0=None, k_max=300):
    n = np.shape(A)[0]
    d = np.diag(A)
    D = (2/w-1)*d
    M = (1/w)*np.diag(d) + np.tril(A, -1)
    N = ((1-w)/w)*np.diag(d) - np.triu(A, 1)
    if y0 is None:
        y = np.zeros(n)
    else:
        y = y0
    y_prev = np.empty(n)
    if mean is None:
        mean = np.zeros(n)
    else:
        mean = A@mean
    def app_MA(v):
        v1 = np.diag(d) @ sp.solve_triangular(M, A @ v)
        return sp.solve_triangular(np.transpose(M), v1)
    B = LinearOperator((n, n), app_MA)
    if d_max is None:
        max_vals, _ = sl.eigs(B, k=1, which='LM')
        d_max = max_vals[0]
        print(d_max)
    if d_min is None:
        min_vals, _ = sl.eigs(B, k=1, sigma=0, which='LM')
        d_min = min_vals[0]
        print(d_min)
    delta = math.pow(((d_max-d_min)/4), 2)
    tau = 2/(d_max+d_min)

    beta = tau
    alpha = 1
    b = 2/alpha-1
    a = (2/tau-1)*b
    k = tau
    z = np.empty(n)
    for i in range(k_max):
        for m in range(n):
            z[m] = np.random.normal(0, b*D[m])
        x = y + sp.solve_triangular(M, z - A@y, lower=True)
        for m in range(n):
            z[m] = np.random.normal(0, a*D[m])
        vv = x - y + sp.solve_triangular(np.transpose(M), z - A@x, lower=False)

        if i==0:
            y_prev = y
            y = alpha*(y + tau*vv)
        else:
            temp = np.copy(y)
            y = y_prev + alpha*(y - y_prev + tau*vv)
            y_prev = temp
        beta = 1/(1/tau - beta*delta)
        alpha = beta/tau
        b = 2*k*(1-alpha)/beta + 1
        a = (2/tau-1) + (b-1)*(1/tau+1/k-1)
        k = beta + (1-alpha)*k
    return y+mean

def findMA(A, w=1):
    D = np.diag(A)
    M = (1/w)*np.diag(D) + np.tril(A, k=-1)
    return sp.inv(M @ np.diag(1/D) @ np.transpose(M)) @ A

b = np.array([1.0, -1.0, 0.0, 2.])

A = np.array([[7., .3, .2, .5],
               [.3, 4, .1, .3],
               [.2, .1, 3, .2],
               [.5, .3, .2, 6]])
target_mean = np.array([3, -1, -2, 6])

#test_size = 200
#test_set = np.empty((test_size, 4))
#test_set2 = np.empty((test_size, 4))
#for x in range(test_size):
#    test_set[x] = SSOR_sampler(A, mean=target_mean)
#    #test_set2[x] = cheby_acc_SSOR_sampler(A, mean=target_mean, d_min=0.251928, d_max=1)
#    test_set2[x] = cheby_acc_SSOR_sampler(A, mean=target_mean)

#print(findMA(A))
#print(np.mean(test_set, axis=0))
#cov = np.cov(np.transpose(test_set))
#precision = sp.inv(cov)
#print(precision)
#print()
#print(np.mean(test_set2, axis=0))
#cov2 = np.cov(np.transpose(test_set2))
#precision2 = sp.inv(cov2)
#print(precision2)
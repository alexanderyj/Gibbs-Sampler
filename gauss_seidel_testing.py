import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as ppt
import gauss_seidel as gs
import math

A = np.array([[5.1, 1.2], 
              [1.2, 8.7]])
b = np.array([20, -13])
c = sp.solve(A, b)
delta = 0.025
x = np.arange(-10, 15, delta)
y = np.arange(-20, 10, delta)
X, Y = np.meshgrid(x, y)
xTAx = A[0][0]*X*X + A[1][0]*Y*X + A[0][1]*X*Y + A[1][1]*Y*Y
Z = 0.5*xTAx - (b[0]*X + b[1]*Y)

#Plots solution to Ax=b by plotting contours of x^T A x - b x

GS = ppt.contour(X, Y, Z, levels=10)

def plotted_gauss_seidel(A, b, x0=None, e=0.01):
    #A = L + D + U, M = L + D, N = -U
    bnorm = sp.norm(b, 2)
    M = np.tril(A)
    if x0==None:
        x0 = np.zeros(b.size)
    r = b - (A @ x0) #r_0
    x_prev = np.copy(x0) #Initially x_0
    x = np.copy(x0)
    while(sp.norm(r, 2) > e*bnorm):
        #print(r)
        x = x + sp.solve_triangular(M, r, lower=True) #x_(k+1) = x_k + M^-1 r_k
        #print(f"{x_prev} to {x}")
        ppt.plot([x[0], x_prev[0]], [x[1],x_prev[1]], 'k')
        r = b - (A @ x) #r_(k+1) = b - A * (x_(k+1))
        x_prev = x
    return x

def plotted_gibbs_sampler(A, mean=None, y0=None, k_max=200, color = 'r'):
    #A is assumed to be symmetric positive definite
    n = np.shape(A)[0]
    if y0==None:
        y0 = np.zeros(n)
    if mean.all()==None:
        mean = np.zeros(n)
    S = np.sqrt(np.diag(A))
    M = np.tril(A)
    N = -1*np.triu(A, 1)
    y = np.copy(y0)
    y_prev = np.copy(y0)

    c_mean = A @ mean
    for k in range(k_max):
        #Noise vector c_k
        c = np.empty(n)
        for x in range(n):
            c[x] = np.random.normal(c_mean[x], S[x])
        y = sp.solve_triangular(M, (N @ y) + c, lower=True)
        ppt.plot([y[0], y_prev[0]], [y[1], y_prev[1]], color)
        y_prev = y
    return y

#plotted_gauss_seidel(A, b)
#for x in range(3):
    #print(plotted_gibbs_sampler(A, c, k_max = 8))

#ppt.show()

A = np.array([
    [ 82,   1, -31,  -4,  25, -12,  -1,   3, -19,   7,   6,  15, -12,   1,   1],
    [  1,  98, -33,  14, -26,  -2,  23,  15,  11,   5,  11,   5,   2, -20, -11],
    [-31, -33,  69,   4,   1,   1, -14, -22,  10,  -3, -21,  -1,  15,   8,   6],
    [ -4,  14,   4,  69,  25, -17,  29, -14,   2,  12,  12,   3, -22,  -4,  12],
    [ 25, -26,   1,  25,  86, -29,  -5, -21, -25,  18,   5,   7, -19,  -1,   2],
    [-12,  -2,   1, -17, -29,  46,  11,  11,   9,   0,  -4,  -4,  11,  -1,  -4],
    [ -1,  23, -14,  29,  -5,  11,  87,  -7,   7,  17,  -5, -11,  -2,   5,  16],
    [  3,  15, -22, -14, -21,  11,  -7,  54,  -3,  12,  11,   0, -16,  -3, -21],
    [-19,  11,  10,   2, -25,   9,   7,  -3,  69,  17,   0,  -5,  20,   8,  13],
    [  7,   5,  -3,  12,  18,   0,  17,  12,  17,  75,  -5,   2,  -4,   4,   9],
    [  6,  11, -21,  12,   5,  -4,  -5,  11,   0,  -5,  82,  -4,  -5,   1,   1],
    [ 15,   5,  -1,   3,   7,  -4, -11,   0,  -5,   2,  -4,  42,  -7,  -8,   3],
    [-12,   2,  15, -22, -19,  11,  -2, -16,  20,  -4,  -5,  -7,  77,  -4,  11],
    [  1, -20,   8,  -4,  -1,  -1,   5,  -3,   8,   4,   1,  -8,  -4,  36,  10],
    [  1, -11,   6,  12,   2,  -4,  16, -21,  13,   9,   1,   3,  11,  10,  76]
])
target_mean = np.array([3, -1, -2, 6, 7, 4, -3, -5, 0, -1, 2, 1, 2, -3, 8])

def SSOR_gibbs_convergence(A, mean=None, y0=None, w=1, k_max=300, test_size = 300, print_cov = False):
    n = np.shape(A)[0]
    D = np.diag(np.diag(A))
    sqrtD = np.sqrt(D)
    M_SOR = (1/w)*D + np.tril(A, -1)
    N_SOR = ((1-w)/w)*D - np.triu(A, 1)
    c = math.sqrt((2/w - 1))
    z = np.empty(n)
    exp_cov = sp.inv(A)

    test_set = np.empty((test_size, n))
    if mean is None:
        mean = np.zeros(n)
    if y0 is None:
        for k in range(test_size):
            test_set[k] = np.zeros(n)
    else:
        for k in range(test_size):
            test_set[k] = y*np.ones(n)

    for k in range(k_max):
        for m in range(test_size):
            for a in range(n):
                z[a] = np.random.normal(0, 1)
            x = sp.solve_triangular(M_SOR, N_SOR@test_set[m] + c*sqrtD@z, lower=True)
            for a in range(n):
                z[a] = np.random.normal(0, 1)
            test_set[m] = sp.solve_triangular(np.transpose(M_SOR), np.transpose(N_SOR)@x + c*sqrtD@z, lower=False)

        test_cov = np.cov(np.transpose(test_set))
        test_mean = np.mean(test_set, axis=0)
        mean_error = sp.norm(test_mean, 2)
        pre_error = sp.norm(test_cov - exp_cov, 2)
        print(f"Iteration {k}: Mean error {mean_error}, Covariance error {pre_error}")
        if(print_cov):
            print(test_cov)
    return test_set+mean

def SSOR_gibbs_convergence_KL_div(A, mean=None, y0=None, w=1, k_max=300, test_size = 300, print_pre = False):
    n = np.shape(A)[0]
    D = np.diag(np.diag(A))
    sqrtD = np.sqrt(D)
    M_SOR = (1/w)*D + np.tril(A, -1)
    N_SOR = ((1-w)/w)*D - np.triu(A, 1)
    c = math.sqrt((2/w - 1))
    z = np.empty(n)
    exp_cov = sp.inv(A)
    test_set = np.empty((test_size, n))
    if mean is None:
        mean = np.zeros(n)
    if y0 is None:
        for k in range(test_size):
            test_set[k] = np.zeros(n)
    else:
        for k in range(test_size):
            test_set[k] = y*np.ones(n)

    for k in range(k_max):
        for m in range(test_size):
            for a in range(n):
                z[a] = np.random.normal(0, 1)
            x = sp.solve_triangular(M_SOR, N_SOR@test_set[m] + c*sqrtD@z, lower=True)
            for a in range(n):
                z[a] = np.random.normal(0, 1)
            test_set[m] = sp.solve_triangular(np.transpose(M_SOR), np.transpose(N_SOR)@x + c*sqrtD@z, lower=False)

        test_cov = np.cov(np.transpose(test_set))
        test_mean = np.mean(test_set, axis=0)
        test_pre = sp.inv(test_cov)
        #pre_error = sp.norm(test_pre - A, 2)
        KL_div = 0.5*(test_mean@test_pre@np.transpose(test_mean) + np.trace(test_pre@exp_cov) - math.log(np.linalg.det(exp_cov)/np.linalg.det(test_cov)) - n)
        print(f"Iteration {k}: KL-divergence {KL_div}")
        if(print_pre):
            print(test_pre)
    return test_set+mean

SSOR_gibbs_convergence(A, target_mean, k_max = 500, test_size=800)
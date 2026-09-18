import numpy as np
import scipy.linalg as sp
import scipy.stats
import matplotlib.pyplot as ppt
import gauss_seidel as gs
import math
from scipy.sparse.linalg import LinearOperator
import scipy.sparse.linalg as sl

A = np.array([[13.2, 1.9], 
              [1.9, 2.1]])
b = np.array([-3, 4])
c = sp.solve(A, b)
delta = 0.025
x = np.arange(-3, 1, delta)
y = np.arange(-1, 6, delta)
X, Y = np.meshgrid(x, y)
xTAx = A[0][0]*X*X + A[1][0]*Y*X + A[0][1]*X*Y + A[1][1]*Y*Y
Z = 0.5*xTAx - (b[0]*X + b[1]*Y)

#Plots solution to Ax=b by plotting contours of x^T A x - b x
ppt.figure(0)
GS = ppt.contour(X, Y, Z, levels=16)

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
#print(plotted_gibbs_sampler(A, c, k_max = 8))
#print(plotted_gibbs_sampler(A, c, k_max = 8, color = 'b'))
#print(plotted_gibbs_sampler(A, c, k_max = 8, color = 'k'))

#ppt.show()

iw = scipy.stats.invwishart(df=220, scale=np.identity(200))
A = iw.rvs()
target_mean = np.zeros(200)

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
        print(f"Iteration {k+1}: Mean error {mean_error}, Covariance error {pre_error}")
        if(k>0):
            ppt.plot([k, k+1], [last_error, pre_error], 'k')
        last_error = pre_error
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

def cheby_acc_gibbs_convergence(A, d_max = None, d_min = None, mean=None, y0=None, w=1, k_max=300, test_size=300, print_cov = False):
    n = np.shape(A)[0]
    d = np.diag(A)
    D = (2/w-1)*d
    M = (1/w)*np.diag(d) + np.tril(A, -1)
    N = ((1-w)/w)*np.diag(d) - np.triu(A, 1)
    y_prev = np.empty((test_size, n))
    exp_cov = sp.inv(A)
    test_set = np.empty((test_size, n))
    for i in range(test_size):
        if y0 is None:
            test_set[i] = np.zeros(n)
        else:
            test_set[i] = y0
    if mean is None:
        mean = np.zeros(n)
    else:
        mean = A@mean
    def app_MA(v):
        v1 = np.diag(d) @ sp.solve_triangular(M, A @ v)
        return sp.solve_triangular(np.transpose(M), v1)
    B = LinearOperator((n, n), app_MA)
    if d_max is None:
        max_vals, _ = sl.eigsh(B, k=1, which='LM')
        d_max = max_vals[0]
        print(d_max)
    if d_min is None:
        min_vals, _ = sl.eigsh(B, k=1, sigma=0, which='LM')
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
        for index in range(test_size):
            for m in range(n):
                z[m] = np.random.normal(0, b*D[m])
            x = test_set[index] + sp.solve_triangular(M, z - A@test_set[index], lower=True)
            for m in range(n):
                z[m] = np.random.normal(0, a*D[m])
            vv = x - test_set[index] + sp.solve_triangular(np.transpose(M), z - A@x, lower=False)

            if i==0:
                y_prev[index] = test_set[index]
                test_set[index] = alpha*(test_set[index] + tau*vv)
            else:
                temp = np.copy(test_set[index])
                test_set[index] = y_prev[index] + alpha*(test_set[index] - y_prev[index] + tau*vv)
                y_prev[index] = temp



        beta = 1/(1/tau - beta*delta)
        alpha = beta/tau
        b = 2*k*(1-alpha)/beta + 1
        a = (2/tau-1) + (b-1)*(1/tau+1/k-1)
        k = beta + (1-alpha)*k
        test_cov = np.cov(np.transpose(test_set))
        test_mean = np.mean(test_set, axis=0)
        mean_error = sp.norm(test_mean, 2)
        pre_error = sp.norm(test_cov - exp_cov, 2)
        print(f"Iteration {i+1}: Mean error {mean_error}, Covariance error {pre_error}")
        if(print_cov):
            print(test_cov)

    return test_set

ppt.figure(1)
SSOR_gibbs_convergence(A, k_max = 300, test_size = 400)
cheby_acc_gibbs_convergence(A, d_max=1, mean=target_mean, k_max = 400, test_size=600)
#ppt.show()
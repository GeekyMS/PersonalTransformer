import numpy as np
from common.grad_check import check_grad


def softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()


def analytic_jacobian_row(p, i):
    # p: the already-computed softmax output vector (cached from forward)
    # returns J[j] = d p_i / d s_j, for every j -- one row of the Jacobian
    T = len(p)
    J = np.zeros(T)
    for j in range(T):
        if i == j:
            J[j] = p[i] * (1 - p[j])   # <- your i==j formula, in terms of p[i] / p[j]
        else:
            J[j] = -1 * p[i] * p[j]   # <- your i!=j formula, in terms of p[i] / p[j]
    return J

def softmax_vjp(p, dP):
    return p * dP - p * (p @ dP.T)

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    s = rng.normal(size=4).astype(np.float64)   # tiny T=4, float64
    p = softmax(s)

    dP = rng.normal(size=4)   # some arbitrary incoming gradient, same length as p
    g = lambda s: np.dot(dP, softmax(s))    # scalar function of s
    analytic = softmax_vjp(p, dP)            # your new collapsed formula

    err = check_grad(g, s, analytic)
    print('VJP max relative error:', err)
    


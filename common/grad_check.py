import numpy as np


def check_grad(f, x, analytic_grad, h=1e-5):
    num_grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index      # coordinate tuple of the current element, any shape

        old = x[i]
        x[i] = old + h
        fp = f(x)               
        x[i] = old - h
        fm = f(x) 
        x[i] = old               # restore — x must be unchanged after the check

        num_grad[i] = (fp - fm) / (2 * h)
        it.iternext()

    rel = np.abs(num_grad - analytic_grad) / (np.abs(num_grad) + np.abs(analytic_grad) + 1e-12)
    return rel.max()

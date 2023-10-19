import numpy as np
import matplotlib.pyplot as plt

from optimization_utils import ObjectiveFunction
from steepest_descent import steepest_descent


def f(x):
    return sum(x**2)


def grad_f(x):
    return 2*x


x0 = 5*np.random.rand(2, )

obj = ObjectiveFunction(f, grad_f)

x1 = np.array([1, 1])

print(obj(x1), obj.grad(x1))

x_iter, f_iter = steepest_descent(obj,
                                  obj.grad,
                                  x0=x0,
                                  alpha=0.01,
                                  epsilon=1e-6)


fig, ax = plt.subplots(1, 1, figsize=(5, 5,))
ax.semilogy(np.arange(1, len(f_iter)+1), f_iter)
ax.set_ylabel('Function value')
ax.set_xlabel('Iterations')

fig.savefig('SD_mwe1.png')

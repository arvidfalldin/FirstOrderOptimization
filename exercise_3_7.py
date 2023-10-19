import numpy as np
import matplotlib.pyplot as plt

from utils.steepest_descent import steepest_descent
"""
Script that tests the number of iterations needed to solve an underdetermined
minimizations problem.
TODO: add analytical expressions for estimates of number of iterations needed
"""


# Choose some dimensions n and d, n should be less than d
n, d = 3, 8

# Seed rng to get reproducible results
np.random.seed(42)

# Construct an underdetermined system (there's probably a better way)
A = np.random.rand(n, n)
Q, __ = np.linalg.qr(A)
R = np.triu(np.random.rand(n, d))
A = Q @ R

b = np.random.rand(n)

# Compute analytical expression for gradient lipschitz
L = np.linalg.norm(A.T @ A, ord=2)
# ...and choose step length accordingly
alpha = 1/L


def f(x):
    return sum((A @ x - b)**2) / n


def grad_f(x):
    return 2*A.T @ (A @ x - b) / n


fig, ax = plt.subplots(1, 1, figsize=(6, 5,))

# If we start at the origin, steepest descent will step
# toward the smallest solution \Tilde{x}.
# Initial gue ss
x0 = np.zeros(d)
x_iter, f_iter = steepest_descent(f,
                                  grad_f,
                                  x0=x0,
                                  alpha=alpha,
                                  max_iter=10000)
ax.semilogy(np.arange(1, len(f_iter)+1), f_iter)

# But if we start somewhere else, we will find other solutions
# #  Uncomment to try ##
# for i in range(20):
#     if i == 0:
#         x0 = np.zeros(d)
#     else:
#         x0 = 10*np.random.rand(d)

#     x_iter, f_iter = steepest_descent(obj,
#                                       obj.grad,
#                                       x0=x0,
#                                       alpha=alpha,
#                                       max_iter=10000)
#     ax.semilogy(np.arange(1, len(f_iter)+1), f_iter)
#     print(f"{f(x_iter[-1]):1.4E}", np.linalg.norm(x_iter[-1]))

ax.set_ylabel('Function value [log scale]')
ax.set_xlabel('Iterations')

fig.savefig('E3_7.png')

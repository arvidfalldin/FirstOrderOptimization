import numpy as np
import matplotlib.pyplot as plt

"""
Script for making the comparison of convergence rates for different
first order steepest descent methods.
"""

from utils.steepest_descent import (steepest_descent,
                                    steepest_descent_with_line_search,
                                    heavy_ball, nestorovs_optimal)


# Objective function settings
mu, L = 0.1, 1
kappa = L/mu
n = 100

# optimization settings
epsilon = 1e-6
kmax = 1000
n_trials = 30

# ## Create a symmetric positive definite matrix
# Init a random square matrix
A = np.random.randn(n, n)

# Get an orthogonal basis of R^n
Q, __ = np.linalg.qr(A)

# Crate a vector of eigenvalues
D = np.random.rand(n)
D = 10**D
Dmin, Dmax = D.min(), D.max()
D = (D - Dmin) / (Dmax - Dmin)
D = mu + D*(L - mu)

# Build matrix with eigenvalue ranging from L to mu
A = Q.T @ np.diag(D) @ Q

# Compute matrix power needed for the line search calcs
A2 = A @ A
A3 = A @ A2


# Create callable handles for all necessary functions
def f(x):
    return 0.5 * (x.T @ A @ x)


def grad_f(x):
    return A @ x


def alpha_sdls(x):
    return (x.T @ A2 @ x) / (x.T @ A3 @ x)


# Params/Settings for the different opt methods:
# Settings for regular gradient descent0
# TODO: Move these below 'trials' dict instead
alpha_sd1 = 2/(mu+L)
sd1_settings = {'f': f, 'grad_f': grad_f, 'epsilon': epsilon,
                'max_iter': kmax, 'alpha': alpha_sd1}

alpha_sd2 = 1/L
sd2_settings = sd1_settings.copy()
sd2_settings['alpha'] = alpha_sd2

# Steepest descent with heavy ball method
sdls_settings = {'f': f, 'grad_f': grad_f, 'epsilon': epsilon,
                 'max_iter': kmax, 'alpha': alpha_sdls}

# Heavy-ball method
alpha_hb = 4 / (np.sqrt(L) + np.sqrt(mu))**2
beta_hb = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
hb_settings = {'f': f, 'grad_f': grad_f, 'epsilon': epsilon,
               'max_iter': kmax, 'alpha': alpha_hb, 'beta': beta_hb}

# Nestorov's optimal method
alpha_no = 1 / L
beta_no = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
no_settings = {'f': f, 'grad_f': grad_f, 'epsilon': epsilon,
               'max_iter': kmax, 'alpha': alpha_no, 'beta': beta_no}


# Init some lists to log the number of iter needed for each method
trials = {
    'SD1': {'algorithm': steepest_descent,
            'settings': sd1_settings,
            'color': 'k',
            'n_iterations': [],
            'plot': None},
    'SD2': {'algorithm': steepest_descent,
            'settings': sd2_settings,
            'color': 'b',
            'n_iterations': [],
            'plot': None},
    'SDLS': {'algorithm': steepest_descent_with_line_search,
             'settings': sdls_settings,
             'color': 'g',
             'n_iterations': [],
             'plot': None},
    'HB': {'algorithm': heavy_ball,
           'settings': hb_settings,
           'color': 'r',
           'n_iterations': [],
           'plot': None},
    'NO': {'algorithm': nestorovs_optimal,
           'settings': no_settings,
           'color': 'm',
           'n_iterations': [],
           'plot': None}
           }

# Create fig, axis to plot from
fig, ax = plt.subplots(1, 1, figsize=(6, 5,))

# Repeat the experiment for n_trials
for i in range(n_trials):
    # Each time we use a new x0
    x0 = np.random.rand(n)

    # Randomize transparancy value for curves
    transparancy = np.random.rand()

    # Run steepest descent with each respective method and plot convergence
    for key, dict in trials.items():
        __, f_iter = dict['algorithm'](**dict['settings'], x0=x0)
        # log the number of iterations needed
        dict['n_iterations'].append(len(f_iter))
        # Plot convergence curve
        dict['plot'] = ax.semilogy(
            np.arange(1, len(f_iter)+1), f_iter, color=dict['color'],
            alpha=transparancy, label=key)

# Compute average number of iterations and print results
handles = []
print("Average number of iterations:")
for key in trials.keys():
    handles.append(trials[key]['plot'][0])
    avg = np.mean(trials[key]['n_iterations'])
    print(f"{key}: {avg:.2f}")
# Add legend and labels to plot before saving figure
ax.legend(handles=handles)
ax.set_ylabel('Function value [log scale]')
ax.set_xlabel('Iterations')
fig.savefig('E4_2.png')

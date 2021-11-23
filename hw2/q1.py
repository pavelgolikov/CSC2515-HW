"""
CSC2515 HW 2-Question 1

Collaborators: Barza Nisar, Yash Kant, Pavel Golikov
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {'family': 'DejaVu Sans',
        'size': 18}

matplotlib.rc('font', **font)

# Plot bias and variance of l2 regularized mean estimator as a function of lambda
def visualize_bias_var():
    mu = 1
    n = 10
    sigma_sq = 9
    lamda = np.arange(0, 10, 0.1)

    bias = (mu**2) * (lamda**2) / (1+lamda)**2
    variance = sigma_sq/(n*(lamda+1)**2)
    expected_sq_err = bias + variance

    plt.figure(figsize=(10,6))
    plt.plot(lamda, variance, 'r', label='Variance', linewidth= 2)
    plt.plot(lamda, bias, 'b', label='Bias', linewidth = 2)
    plt.plot(lamda, expected_sq_err, 'g', label='Expected Sq Err', linewidth=2)
    plt.xlabel('lambda')
    plt.grid()
    plt.legend()
    plt.savefig('bias_var.png')
    plt.show()


# Plot bias and variance of l2 regularized mean estimator as a function of lambda
visualize_bias_var()






#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.integrate import quad


def integrand(theta, y_star, y, alpha, N, beta):
    ppd = stats.poisson.pmf(y_star, theta) * stats.gamma.pdf(theta, a=(y + alpha), scale=1/(N + beta))
    return ppd


def main():
    # Example 2.1:  Poisson Rate --- Posterior Inference ###
    y = 35
    N = 0.402
    # Gamma prior with mean of 16 per year, but relatively noninformative:
    alpha = 1.6
    beta = 0.1

    # Posterior mean and standard deviation (based on analytical solution)

    print(f'posterior mean : {(y+alpha) / (N+beta)}')
    print(f'posterios SD {np.sqrt((y+alpha)/(N+beta)**2)}')

    # 95% equal-tailed credible interval (based on analytical solution)
    print(f'95% equal-tailed credible interval (based on analytical solution) : '
          f'{stats.gamma.ppf([0.025, 0.975], a=(y+alpha),scale=1/(N+beta))}')

    fig, ax = plt.subplots()
    x = np.arange(0, 150)
    ax.plot(x, stats.gamma.pdf(x, a =(y + alpha), scale=1 /(N + beta)))
    ax.set_ylabel('PDF')
    ax.set_xlabel(r'$\theta$')
    ax.set_title("Posterior: Detection rate per year")
    plt.show()

    # Posterior prediction: In one year of the next run (O4), how many
    #  "confirmed" detections to expect?

    # Using numerical integration
    y_star = np.arange(0, 150)  # plot domain

    ppp = [quad(integrand, 0, np.inf, args=(i, y, alpha, N, beta), epsabs=1e-15)[0] for i in y_star]

    fig, ax = plt.subplots()
    ax.vlines(y_star, 0, ppp)
    ax.set_ylabel('Post.Pred.Prob')
    ax.set_xlabel(r'$y^{*}$')
    ax.set_title("Posterior: Detection rate per year")
    plt.show()
    print(1)


if __name__ == "__main__":
    main()

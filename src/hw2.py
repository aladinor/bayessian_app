#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.special import comb, gamma, factorial


def integrand(theta, y_star, n_star, y, n):
    ppd = stats.binom.pmf(y_star, n_star, theta) * stats.beta.pdf(theta, y + 1, n - y + 1)
    return ppd


def main():
    m = 5
    theta = 0.2
    x = np.arange(0, 100)
    neg_bin = stats.nbinom.pmf(x, m, theta)
    fig, ax = plt.subplots()
    ax.vlines(x, 0, neg_bin)
    plt.show()
    print(1)

    alpha = 6
    beta = 25
    equ_tled = stats.beta(alpha, beta).ppf([0.025, 0.975])
    print(f"equaly tiled values are : {equ_tled[0]} and {equ_tled[1]}")

    #  Posterior prob. that theta < 20%   (at least 30% own pets)

    post_prob = stats.beta(alpha, beta).cdf(0.2)
    print(f"posterior probalibilty less or equal to  20%  is {post_prob}")


    a = 6
    b = 25
    x = np.linspace(stats.beta.ppf(0.01, alpha, beta), stats.beta.ppf(0.99, alpha, beta), 100)
    fig, ax = plt.subplots()
    ax.plot(x, stats.beta.pdf(x, alpha, beta))
    plt.show()

    y_star = np.arange(0, 20)
    ppd = []
    _ppd = []
    for i in range(len(y_star)):
        ppd.append(comb(y_star[i] + 4, y_star[i]) * ((gamma(31) / (gamma(6) * gamma(25))) * (gamma(11) * gamma(y_star[i] + 25) / gamma(36))))
        _ppd.append(comb(y_star[i] + 4, y_star[i]) * (factorial(30) / (factorial(5) * factorial(24))) * (factorial(10) * factorial(y_star[i] + 24))/ factorial(35))

    print(1)


if __name__ == "__main__":
    main()

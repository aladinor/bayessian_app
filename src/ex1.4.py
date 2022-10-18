#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.special import binom, comb
from scipy.integrate import quad


def integrand(theta, y_star, n_star, y, n):
    ppd = stats.binom.pmf(y_star, n_star, theta) * stats.beta.pdf(theta, y + 1, n - y + 1)
    return ppd


def main():
    n = 70
    y = 12
    n_star = 10
    y_star = np.arange(0, n_star)

    #   Method 1: Using Explicit Formula

    ppd = comb(n_star, y_star) * (n + 1) * comb(n, y) / ((n + n_star + 1) * comb(n + n_star, y + y_star))
    fig, ax = plt.subplots()
    ax.vlines(y_star, 0, ppd)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Post.pred.probs')
    ax.set_xlabel(r'$y^{*}$')
    ax.set_title("Explicit Formula")
    plt.show()

    # Method 2: Using Numerical Integration

    ppd = []
    for i in range(n_star):
        ppd.append(quad(integrand, 0, 1, args=(y_star[i], n_star, y, n))[0])
    fig, ax = plt.subplots()
    ax.vlines(y_star, 0, ppd)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Post.pred.probs')
    ax.set_xlabel(r'$y^{*}$')
    ax.set_title("Numerical Integration")
    plt.show()

    pass


if __name__ == "__main__":
    main()

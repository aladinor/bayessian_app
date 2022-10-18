#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb, beta


def post_pred_dens(y_star, n_star, alpha, _beta, n, y):
    print(1)
    ppd = comb(n_star, y_star) * beta(y_star + y + alpha, n_star - y_star + n - y + _beta) / \
          beta(y + alpha, n - y + _beta)
    return ppd


def main():
    #  Example 1.6:  Population Proportion --- Posterior Predictive Checks ###
    n = 70
    y = 12
    n_star = n
    ppd = [post_pred_dens(y_star=i, n_star=n, alpha=1, _beta=1, n=n, y=y) for i in np.arange(n_star)]
    # posterior predictive density of the data, given prior Beta(alpha,beta)
    fig, ax = plt.subplots()
    ax.vlines(np.arange(n_star), 0, ppd)
    ax.set_ylabel('Density')
    ax.scatter(y, 0, c='r', lw=2)
    ax.set_xlabel(r'$y^{*}$')
    ax.set_title("Data and PPD")
    plt.show()

    # What if we use the misinformative prior (alpha=100,beta=1)?

    ppd = [post_pred_dens(y_star=i, n_star=n, alpha=100, _beta=1, n=n, y=y) for i in np.arange(n_star)]
    # posterior predictive density of the data, given prior Beta(alpha,beta)
    fig, ax = plt.subplots()
    ax.vlines(np.arange(n_star), 0, ppd)
    ax.set_ylabel('Density')
    ax.scatter(y, 0, c='r', lw=2)
    ax.set_xlabel(r'$y^{*}$')
    ax.set_title("Data and PPD")
    plt.show()
    print('Done!')


if __name__ == "__main__":
    main()

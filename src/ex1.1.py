#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def hist_prior(theta):
    if theta < 0.15:
        return 0.25 / 0.15
    elif theta < 0.25:
        return 0.30 / 0.1
    elif theta < 0.35:
        return 0.25 / 0.1
    elif theta < 0.45:
        return 0.05 / 0.1
    else:
        return 0.15 / 0.55


def main():
    #  Example 1.1:  Population Proportion: Likelihood, Priors, Posteriors
    n = 70

    #  (a) Binomial probability models --- for example ...

    theta = np.array([0.05, 0.1, 0.3, 0.5])
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    x = np.arange(0, n)
    for i, ax in enumerate(axs.flat):
        bn = stats.binom.pmf(x, n, theta[i])
        ax.vlines(x, 0, bn)
        ax.set_xlabel(f"$\\theta = {theta[i]}$")
        ax.set_ylabel(f"$Density$")
    plt.show()

    #  (b) Binomial model likelihood

    dx = 1 / 1000
    theta_grid = np.arange(0, 1, dx)
    y = 12

    fig, axs = plt.subplots(1, 1)
    lkhd = stats.binom.pmf(y, n, theta_grid)
    axs.plot(theta_grid, lkhd)
    axs.set_ylabel(r'$PMF$')
    axs.set_xlabel(r'$\theta$')
    axs.set_title(r"$Likekihood \ for \  y=12$")
    plt.show()

    #  (c) Prior densities

    # flat
    flat_prior = stats.uniform.pdf(theta_grid, 0, 1)

    # beta (matched to mean and standard deviation of guesses)
    mean = 0.236
    sd = 0.147
    alpha = mean * (mean * (1 - mean) / sd ** 2 - 1)
    beta = alpha * (1 / mean - 1)
    beta_prior = stats.beta.pdf(theta_grid, alpha, beta)

    #  five-point (discrete)

    pt_prior = dict(theta=[0.05, 0.15, 0.25, 0.35, 0.5],
                    density=[0.10, 0.30, 0.30, 0.15, 0.15])

    # histogram
    hist_pr = np.vectorize(hist_prior)(theta_grid)

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    axs[0][0].plot(theta_grid, flat_prior)
    axs[0][0].set_title(r"$Flat \  pior$")
    axs[0][0].set_ylim(0, 2)
    axs[0][0].set_xlabel(r'$\theta$')

    axs[0][1].plot(theta_grid, beta_prior)
    axs[0][1].set_title(r"$Beta$")
    axs[0][1].set_xlabel(r'$\theta$')

    axs[1][0].vlines(pt_prior['theta'], 0, pt_prior['density'])
    axs[1][0].set_title(r"$Discrete$")
    axs[1][0].set_xlabel(r'$\theta$')
    axs[1][0].set_ylim(0, 1)

    axs[1][1].plot(theta_grid, hist_pr)
    axs[1][1].set_title(r"$Histogram$")
    axs[1][1].set_xlabel(r'$\theta$')
    plt.show()

    # (d) Posterior densities

    # flat prior
    post_unscaled = stats.binom.pmf(y, n, theta_grid) * flat_prior
    posterior_flat = post_unscaled / np.sum(post_unscaled * dx)

    # beta prior
    post_unscaled = stats.binom.pmf(y, n, theta_grid) * beta_prior
    posterior_beta = post_unscaled / np.sum(post_unscaled * dx)

    # five-point (discrete) prior
    post_unscaled = stats.binom.pmf(y, n, pt_prior['theta']) * pt_prior['density']
    post_discrete = post_unscaled / np.sum(post_unscaled)

    # histogram prior
    post_unscaled = stats.binom.pmf(y, n, theta_grid) * hist_pr
    post_histogram = post_unscaled / np.sum(post_unscaled * dx)

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    axs[0][0].plot(theta_grid, posterior_flat)
    axs[0][0].set_title(r"$Flat \  post$")
    axs[0][0].set_xlabel(r'$\theta$')

    axs[0][1].plot(theta_grid, posterior_beta)
    axs[0][1].set_title(r"$Beta \  post$")
    axs[0][1].set_xlabel(r'$\theta$')

    axs[1][0].vlines(pt_prior['theta'], 0, post_discrete)
    axs[1][0].set_title(r"$Discrete \  post$")
    axs[1][0].set_xlabel(r'$\theta$')

    axs[1][1].plot(theta_grid, post_histogram)
    axs[1][1].set_title(r"$Histogram \  post$")
    axs[1][1].set_xlabel(r'$\theta$')
    plt.show()
    print(1)


if __name__ == "__main__":
    main()

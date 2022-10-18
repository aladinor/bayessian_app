#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd


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
    n = 70
    y = 12

    # Example 1.5:  Population Proportion --- Sensitivity Analysis ###
    # Consider the same four priors as in Example 1.1
    # and compute (1) posterior mean, and (2) prob that theta >= 0.3

    dtheta = 1 / 1000  # theta grid spacing
    theta_grid = np.arange(0, 1, dtheta)
    post_means = np.zeros(4)
    post_probs = np.zeros(4)

    # flat (uniform) prior

    flat_prior = stats.uniform.pdf(theta_grid, 0, 1)
    post_unscaled = stats.binom.pmf(y, n, theta_grid) * flat_prior
    posterior = post_unscaled / sum(post_unscaled * dtheta)
    post_means[0] = sum(theta_grid * posterior * dtheta)
    post_probs[0] = sum(posterior[theta_grid >= 0.3] * dtheta)

    # beta prior (matched to mean and standard deviation of guesses)

    mean_ = 0.236
    sd = 0.147

    alpha = mean_ * (mean_ * (1 - mean_) / sd ** 2 - 1)
    beta = alpha * (1 / mean_ - 1)

    beta_prior = stats.beta.pdf(theta_grid, alpha, beta)
    post_unscaled = stats.binom.pmf(y, n, theta_grid) * beta_prior
    posterior = post_unscaled / sum(post_unscaled * dtheta)

    post_means[1] = sum(theta_grid * posterior * dtheta)
    post_probs[1] = sum(posterior[theta_grid >= 0.3] * dtheta)

    # five-point prior (discrete)

    pt_prior = dict(theta=np.array([0.05, 0.15, 0.25, 0.35, 0.5]),
                    density=np.array([0.10, 0.30, 0.30, 0.15, 0.15]))

    pt_post_unscaled = stats.binom.pmf(y, n, pt_prior['theta']) * pt_prior['density']
    pt_posterior = pt_post_unscaled / sum(pt_post_unscaled)

    post_means[2] = sum(pt_prior['theta'] * pt_posterior)
    post_probs[2] = sum(pt_posterior[pt_prior['theta'] >= 0.3])

    # histogram prior

    post_unscaled = stats.binom.pmf(y, n, theta_grid) * (np.vectorize(hist_prior)(theta_grid))
    posterior = post_unscaled / sum(post_unscaled * dtheta)
    post_means[3] = sum(theta_grid * posterior * dtheta)
    post_probs[3] = sum(posterior[theta_grid >= 0.3] * dtheta)

    df = pd.DataFrame(np.stack([post_means, post_probs]), columns=["Flat", "Beta", "Five-Point", "Histogram"],
                      index=['post_measn', 'post_probs']).T


if __name__ == "__main__":
    main()

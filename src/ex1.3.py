#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar


def interval_width(x, n, y):
    _res = stats.beta(y+1, n-y+1).ppf(0.95) - stats.beta(y+1, n-y+1).ppf(x)
    return _res.min()


def main():
    #  Example 1.3:  Population Proportion --- Bayesian Methods ###

    y = 12
    n = 70
    equ_tled = stats.beta(y+1, n-y+1).ppf([0.025, 0.975])
    print(f"equaly tiled values are : {equ_tled[0]} and {equ_tled[1]}")

    # Approximate 95% HPD credible interval under Uniform(0,1) prior
    x0 = np.array([0, 0.05])
    lower_area = minimize_scalar(interval_width, bounds=(0, 0.05), args=(n, y))

    # (b) Posterior Probabilities (of hypotheses)

    # Posterior prob. that theta >= 0.3   (at least 30% own pets)
    # under Uniform(0,1) prior:

    post_prob = 1 - stats.beta(y + 1, n - y + 1).cdf(0.3)
    print(f"posterior probalibilty at least 30% own pets is {post_prob}")


if __name__ == "__main__":
    main()

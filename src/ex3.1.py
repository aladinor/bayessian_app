#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.integrate import quad


def _theta(theta):
    return theta


def likelihood(theta, y, n):
    return theta ** y * (1 - theta) ** (n - y)


def prior(theta):
    return stats.beta.pdf(theta, 0.5, 0.5)


def _num(theta, y, n):
    return _theta(theta) * prior(theta) * likelihood(theta, y, n)


def _den(theta, y, n):
    return prior(theta) * likelihood(theta, y, n)


def main():
    # Example 3.1:  Population Proportion: Numerical Integration ###

    n = 70
    y = 12
    # Define likelihood and prior (up to proportionality)

    # Compute posterior expectation of theta

    numerator = quad(_num, 0, 1, args=(y, n))[0]
    print(f'numerator: {numerator}')
    denominator = quad(_den, 0, 1, args=(y, n))[0]
    print(f'denominator: {denominator}')
    print(numerator / denominator)
    print(1)


if __name__ == "__main__":
    main()

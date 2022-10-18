#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.integrate import quad, dblquad


def _theta(theta):
    return theta


def likelihood(theta1, theta2, y1, n1, y2, n2):
    return theta1 ** y1 * (1 - theta1) ** (n1 - y1) * theta2 ** y2 * (1 - theta2) ** (n2 - y2)


def prior(theta1, theta2):
    return stats.beta.pdf(theta1, 0.5, 0.5) * stats.beta.pdf(theta2, 0.5, 0.5)


def integrand(theta1, theta2, y1, n1, y2, n2):
    return likelihood(theta1, theta2, y1, n1, y2, n2) * prior(theta1, theta2)


def inner_int_num(theta2, y1, n1, y2, n2):
    return quad(integrand, 0, theta2, args=(theta2, y1, n1, y2, n2))[0]


def inner_int_den(theta2, y1, n1, y2, n2):
    return quad(integrand, 0, 1, args=(theta2, y1, n1, y2, n2))[0]


def main():
    # Example 3.2:  Comparing Population Proportions: Numerical Integration ###

    n1 = 19
    y1 = 8
    n2 = 51
    y2 = 25
    eps = 1e-20
    # Define likelihood and prior (up to proportionality)

    # Compute posterior expectation of theta

    numerator = quad(inner_int_num, 0, 1, args=(y1, n1, y2, n2))[0]
    print(f'numerator: {numerator}')
    denominator = quad(inner_int_den, 0, 1, args=(y1, n1, y2, n2))[0]
    print(f'denominator: {denominator}')
    print(numerator / denominator)
    print(1)


if __name__ == "__main__":
    main()

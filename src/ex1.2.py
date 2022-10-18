#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy import stats


def main():
    y = 12
    n = 70
    res = stats.binomtest(y, n, p=0.3, alternative='less')
    print(f"p value = {res.pvalue} and p_succces = {res.proportion_estimate}")


if __name__ == "__main__":
    main()

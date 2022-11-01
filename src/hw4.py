import scipy.stats as ss
import numpy as np
import pandas as pd


def main():
    y = 20
    Nsim = 100000
    rs = np.zeros(Nsim)
    thetas = np.zeros(Nsim)
    rs[0] = 10
    n0 = rs[0] + 20
    thetas[0] = ss.beta.rvs(y + 1, n0 - y + 1)

    for i in range(1, Nsim):
        n_i = rs[i - 1] + 20
        thetas[i] = ss.beta.rvs(a=(y + 1), b=(n_i - y + 1))
        rs[i] = ss.nbinom.rvs(y+1, 0.9 * thetas[i - 1] + 0.1)

    print(1)
    pass


if __name__ == "__main__":
    main()
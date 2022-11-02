import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    df = pd.read_csv('../data/airlineraccident.dat', sep='\s\s+', engine='python')
    df['No. Flights'] = df['No. Flights'].apply(lambda x: float(x.split(' ')[0]))
    ys = df['Events']
    Ns = df['No. Flights']

    alpha = beta = 0.001
    Nsim = 100000
    gammas = np.zeros(Nsim)
    lambdavecs = np.zeros([Nsim, len(df)])

    # Initialize
    lambdavecs[0, :] = 1
    gammas[0] = 1

    for i in range(1, Nsim):
        lambdavecs[i, :] = ss.gamma.rvs(a=ys + 1, scale=1 / (Ns + gammas[i-1]))
        gammas[i] = ss.gamma.rvs(a=20 + alpha, scale=1 / (np.sum(lambdavecs[i, :]) + beta))

    fig, ax = plt.subplots()
    ax.plot(gammas)
    ax.set_xlabel('index')
    ax.set_ylabel('gammas')
    plt.show()

    ### Posterior inference ...
    # Post. mean and std. dev. of "avg" event rate (1/gamma)

    print(f'mean = {np.mean(1 / gammas): .3f}')
    print(f'sd = {np.std(1 / gammas):3f}')

    # 95% credible interval for "avg" event rate (1/gamma)
    print(np.quantile(1 / gammas, [0.025, 0.975]))
    # average of sample rates, for comparison
    print(np.mean(ys/Ns))

    fig, ax = plt.subplots()
    ax.boxplot(lambdavecs)
    ax.set_yscale('log')
    ax.scatter(np.arange(1, len(df) + 1), ys/Ns, c='r', facecolor="None")
    plt.show()

    # Simulate from post. pred. distribution for a "new" lambda
    lambdanews = ss.expon.rvs(gammas, size=Nsim)
    fig, ax = plt.subplots()
    ax.hist(lambdanews)
    plt.show()
    print(np.quantile(lambdanews, [0.025, 0.975]))
    print(1)
    pass


if __name__ == "__main__":
    main()

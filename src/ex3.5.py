import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs):
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins,
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)


def main():
    # Example 3.4:  Gibbs Sampler for Semi-Conjugate Prior (Normal Sample)
    n = 24
    ybar = 7.8730
    s = 0.05353
    mu0 = 7.9876
    sigma0_2 = 0.0003125
    alpha = 2.5
    beta = 0.0009375

    # Run Gibbs Sampler ...
    Nsim = 100000
    mus = np.zeros(Nsim)
    sigma_2s = np.zeros(Nsim)

    # initialize
    mus[0] = 8
    a = n / 2 + alpha
    sc = (((n - 1) * s ** 2 + n * (ybar - mus[0]) ** 2) / 2 + beta)
    sigma_2s[0] = 1 / ss.gamma.rvs(a=a, scale=(1 / sc))

    for i in range(1, Nsim):

        mean = (n * ybar / sigma_2s[i - 1] + mu0 / sigma0_2) / (n / sigma_2s[i - 1] + 1 / sigma0_2)
        sd = np.sqrt(1 / (n / sigma_2s[i - 1] + 1 / sigma0_2))
        mus[i] = ss.norm.rvs(mean, sd)

        a = n / 2 + alpha
        sse = ((n - 1) * s ** 2 + n * (ybar - mus[i]) ** 2)
        sigma_2s[i] = 1 / ss.gamma.rvs(a, scale=1 / (sse/2 + beta))

    # Posterior inference ...

    print(f'Posterior mean = {np.mean(mus)}')
    print(f'Posterior standard deviation = {np.std(mus)}')
    # 95% credible interval for mu (equal-tailed)
    print(f'95% credible interval for mu (equal-tailed) = {np.quantile(mus, [0.025, 0.975])}')
    # compare posterior mean and interval with ybar
    print(f'ybar = {ybar}')
    # Post. prob. of H0: mu >= 7.9379 (minimum legal weight)
    print(f'Post. prob. of H0: mu >= 7.9379 (minimum legal weight) = {np.mean(mus > 7.9379)}')
    # # Posterior mean of sigma^2
    print(f'Posterior mean of sigma^2 = {np.mean(sigma_2s)}')
    #  95% credible interval for sigma^2 (equal-tailed)
    print(f'95% credible interval for sigma^2 (equal-tailed) = {np.quantile(sigma_2s, [0.025, 0.975])}')
    # s ^ 2  # compare posterior mean and interval with s^2
    print(f"posterior mean and interval with s^2 = {s ** 2}")
    # Visualizing the first few steps of the "sample path" ...
    maxit = 20  # number of iterates to show

    fig, ax = plt.subplots()
    ax.plot(mus[:maxit], sigma_2s[:maxit])
    ax.set_title('Joint PDF (informative Semi-Conjugate Prior)')
    ax.set_ylabel(r"$\sigma^2$")
    ax.set_xlabel(r"$\mu$")
    plt.show()

    xx, yy, zz = kde2D(mus, sigma_2s, 1.0)

    fig, ax = plt.subplots()
    ax.pcolormesh(xx, yy, zz)
    ax.scatter(mus, sigma_2s, s=2, facecolor='white')

    print(1)

    pass


if __name__ == "__main__":
    main()
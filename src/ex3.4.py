import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt


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
    maxit = 30  # number of iterates to show

    fig, ax = plt.subplots()
    ax.quiver(mus[:maxit -1], sigma_2s[:maxit -1],
              mus[1:maxit] - mus[:maxit-1],
              sigma_2s[1:maxit] - sigma_2s[:maxit-1],
              scale_units='xy', angles='xy', scale=1, width=0.005)
    ax.set_title('Start of sample path')
    ax.set_ylabel(r"$\sigma^2$")
    ax.set_xlabel(r"$\mu$")
    ax.scatter(mus[:maxit], sigma_2s[:maxit])
    plt.show()

    xmin, xmax = 7.88, 7.94
    ymin, ymax = 0, 0.01
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([mus, sigma_2s])
    kernel = ss.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig, ax = plt.subplots()
    cs = ax.contour(xx, yy, f)
    ax.clabel(cs, inline=True, fontsize=5)
    ax.set_ylabel(r"$\sigma^2$")
    ax.set_xlabel(r"$\mu$")
    ax.set_title('Joint PDF (Informative Semi-Conjugate Prior)')
    plt.show()


if __name__ == "__main__":
    main()
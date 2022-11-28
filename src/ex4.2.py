import arviz as az
import matplotlib.pyplot as plt
import pyjags as pj
import pandas as pd
import numpy as np


def main():
    df = pd.read_csv('../data/ex4.2data.txt', sep='  ', header=None)
    data = {'y': np.ma.masked_invalid(df.values)}
    inits = [{'mu': 1500, 'tausqW': 0.001, 'sigmaB': 100},
             {'mu': 3000, 'tausqW': 1, 'sigmaB': 1},
             {'mu': 0, 'tausqW': 0.000001, 'sigmaB': 10000}]

    jags_model_string = '''
        data {
          dimy <- dim(y)
          batches <- dimy[1]
          samples <- dimy[2]
        }
        
        model {
          for (i in 1:batches) {
        
            for (j in 1:samples) {
              y[i,j] ~ dnorm(alpha[i], tausqW)
            }
        
            alpha[i] ~ dnorm(mu, 1/sigmasqB)
          }
        
          mu ~ dnorm(0, 0.000001)
        
          tausqW ~ dgamma(0.001, 0.001)
          sigmaB ~ dexp(0.001)
        
          sigmasqW <- 1 / tausqW
          sigmasqB <- sigmaB^2
        
          rho <- sigmasqB / (sigmasqB + sigmasqW)
        }
        '''
    jags_model \
        = pj.Model(code=jags_model_string,
                   init=inits,
                   data=data,
                   chains=3,
                   )

    samples_1 = jags_model.sample(iterations=1000, vars=["alpha", "mu", "sigmasqW", "sigmasqB", "rho"])
    idata = az.from_pyjags(samples_1)

    az.plot_trace(idata, var_names=["alpha", "mu", "sigmasqW", "sigmasqB", "rho"])
    plt.tight_layout()
    plt.show()

    # Autocorrelation plots for chain # 1
    az.plot_autocorr(idata.posterior.sel(chain=1), var_names=["alpha", "mu", "sigmasqW", "sigmasqB", "rho"])
    plt.tight_layout()
    plt.show()

    samples_2 = jags_model.sample(iterations=2000, vars=["alpha", "mu", "sigmasqW", "sigmasqB", "rho"])
    idata2 = az.from_pyjags(samples_2)
    az.plot_trace(idata2, var_names=["alpha", "mu", "sigmasqW", "sigmasqB", "rho"])
    plt.tight_layout()
    plt.show()

    samples_3 = jags_model.sample(iterations=102000, vars=["alpha", "mu", "sigmasqW", "sigmasqB", "rho"])
    idata3 = az.from_pyjags(samples_3)
    az.plot_trace(idata3, var_names=["alpha", "mu", "sigmasqW", "sigmasqB", "rho"])
    plt.tight_layout()
    plt.show()

    func_dict = {
        "Mean": np.mean,
        "SD": np.std,
        "SE": lambda x: np.std(x) / np.sqrt(np.size(x)),
        "2.5%": lambda x: np.percentile(x, 2.5),
        "25%": lambda x: np.percentile(x, 25),
        "50%": lambda x: np.percentile(x, 50),
        "75%": lambda x: np.percentile(x, 75),
        "97.5%": lambda x: np.percentile(x, 97.5),
    }
    # Output summary
    summary = az.summary(idata3.posterior.sel(draw=slice(60000, 100000))[["alpha", "mu", "sigmasqW", "sigmasqB",
                                                                          "rho"]],
                         stat_funcs=func_dict)
    print(summary)
    az.plot_trace(idata3.posterior.sel(draw=slice(60000, 100000)), var_names=["alpha", "mu", "sigmasqW", "sigmasqB",
                                                                              "rho"],
                  figsize=(12, 8))
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(idata3.posterior.sel(draw=slice(60000, 100000)).sigmasqW,
               idata3.posterior.sel(draw=slice(60000, 100000)).sigmasqB, s=0.1, fc='k')
    ax.set_xlabel('sigmasqB')
    ax.set_ylabel('sigmasqW')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()
    print(1)


if __name__ == "__main__":
    main()

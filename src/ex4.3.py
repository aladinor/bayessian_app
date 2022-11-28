import arviz as az
import matplotlib.pyplot as plt
import pyjags as pj
import pandas as pd
import numpy as np


def main():
    df = pd.read_csv('../data/ex4.3data.txt', sep='  ', header=None)
    ages = np.array([8, 15, 22, 29, 36])
    data = {'Y': np.ma.masked_invalid(df.values),
            'X': np.ma.masked_invalid(ages),
            'Xbar': np.ma.masked_invalid(np.mean(ages))}
    inits = [{'tausq.y': 1, 'beta1': 0, 'beta2': 0, 'sigma.alpha1': 1, 'sigma.alpha2': 1},
             {'tausq.y': 100, 'beta1': 100, 'beta2': 100, 'sigma.alpha1': 0.1, 'sigma.alpha2': 0.1},
             {'tausq.y': 0.01, 'beta1': -100, 'beta2': -100, 'sigma.alpha1': 10, 'sigma.alpha2': 10}]

    jags_model_string = '''
      data {
          dim.Y <- dim(Y)
        }
        
        model {
          for(i in 1:dim.Y[1]) {
        
            for(j in 1:dim.Y[2]) {
              Y[i,j] ~ dnorm(mu[i,j], tausq.y)
              mu[i,j] <- alpha[i,1] + alpha[i,2] * (X[j] - Xbar)
            }
        
            alpha[i,1] ~ dnorm(beta1, 1 / sigma.alpha1^2)
            alpha[i,2] ~ dnorm(beta2, 1 / sigma.alpha2^2)
          }
          
          tausq.y ~ dgamma(0.001, 0.001)
          sigma.y <- 1 / sqrt(tausq.y)
        
          beta1 ~ dnorm(0.0, 1.0E-6)
          beta2 ~ dnorm(0.0, 1.0E-6)
          sigma.alpha1 ~ dexp(0.001)
          sigma.alpha2 ~ dexp(0.001)
        }
        '''
    jags_model \
        = pj.Model(code=jags_model_string,
                   init=inits,
                   data=data,
                   chains=3,
                   )

    samples_1 = jags_model.sample(iterations=1000, vars=["beta1", "beta2", "sigma.y", "sigma.alpha1", "sigma.alpha2"])
    idata = az.from_pyjags(samples_1)

    az.plot_trace(idata, var_names=["beta1", "beta2", "sigma.y", "sigma.alpha1", "sigma.alpha2"])
    plt.tight_layout()
    plt.show()

    # Autocorrelation plots for chain # 1
    az.plot_autocorr(idata.posterior.sel(chain=1), var_names=["beta1", "beta2", "sigma.y", "sigma.alpha1",
                                                              "sigma.alpha2"])
    plt.tight_layout()
    plt.show()

    samples_2 = jags_model.sample(iterations=11000, vars=["beta1", "beta2", "sigma.y", "sigma.alpha1",
                                                          "sigma.alpha2"])
    idata2 = az.from_pyjags(samples_2)
    az.plot_trace(idata2, var_names=["beta1", "beta2", "sigma.y", "sigma.alpha1",
                                     "sigma.alpha2"])
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
    summary = az.summary(idata2.posterior.sel(draw=slice(4000, 11000))[["beta1", "beta2", "sigma.y", "sigma.alpha1",
                                                                       "sigma.alpha2"]],
                         stat_funcs=func_dict)
    print(summary)

    az.plot_posterior(idata2, var_names=["beta1", "beta2", "sigma.y", "sigma.alpha1", "sigma.alpha2"])
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(idata2.posterior.sel(draw=slice(4000, 10000)).beta1,
               idata2.posterior.sel(draw=slice(4000, 10000)).beta2, s=0.1, fc='k')
    ax.set_xlabel('beta1')
    ax.set_ylabel('beta2')
    plt.show()
    print(1)


if __name__ == "__main__":
    main()

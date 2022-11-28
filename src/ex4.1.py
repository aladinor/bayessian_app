import arviz as az
import matplotlib.pyplot as plt
import pyjags as pj
import pandas as pd
import numpy as np


def main():
    df = pd.read_csv('../data/ex4.1data.txt', header=0, skiprows=4, sep='  ', na_values='NA')
    data = {'y': np.ma.masked_invalid(df['y'].values), 'x': np.ma.masked_invalid(df['x'].values)}
    inits = [{'beta1': 0, 'beta2': 0},
             {'beta1': 1, 'beta2': 1},
             {'beta1': -1, 'beta2': -1}]

    jags_model_string = '''
        data {
          xmean <- mean(x[1:(length(x)-1)])
          for(i in 1:length(x)) {
            xcent[i] <- x[i] - xmean
          }
        }
        model {
          for(i in 1:length(y)) {
            y[i] ~ dpois(lambda[i])
            log(lambda[i]) <- beta1 + beta2 * xcent[i]
          }
        
          beta1 ~ dnorm(0, 0.0001)
          beta2 ~ dnorm(0, 0.0001)
        
          beta2.gt.0 <- beta2 > 0
        }
        '''
    jags_model \
        = pj.Model(code=jags_model_string,
                   init=inits,
                   data=data,
                   chains=3,
                   )

    samples_1 = jags_model.sample(iterations=1000, vars=["beta1", "beta2", "lambda", "y", "beta2.gt.0"])
    idata = az.from_pyjags(samples_1)

    az.plot_trace(idata, var_names=["beta1", "beta2", "lambda", "y", "beta2.gt.0"])
    plt.tight_layout()
    plt.show()

    samples_2 = jags_model.sample(iterations=100000, vars=["beta1", "beta2", "lambda", "y", "beta2.gt.0"])
    idata2 = az.from_pyjags(samples_2)
    az.plot_trace(idata2, var_names=["beta1", "beta2", "lambda", "y", "beta2.gt.0"])
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
    summary = az.summary(idata2.posterior.sel(draw=slice(60000, 100000))[["beta1", "beta2",
                                                                          "lambda", "y", "beta2.gt.0"]],
                         stat_funcs=func_dict)
    print(summary)
    az.plot_posterior(idata2.posterior.sel(draw=slice(60000, 100000)).sel(y_dim_0=13), var_names=["beta1", "beta2",
                                                                                                 "y", "beta2.gt.0"],
                      figsize=(12, 4))
    plt.show()
    plt.tight_layout()
    print(1)


if __name__ == "__main__":
    main()

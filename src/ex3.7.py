import arviz as az
import matplotlib.pyplot as plt
import pyjags as pj
import pandas as pd
import numpy as np


def main():
    df = pd.read_csv('../data/ex3.7data.txt')
    data = {'y': df['y'].values}
    inits = [{'mu': 5, 'tausq': 1.5},
             {'mu': 0, 'tausq': 15},
             {'mu': 10, 'tausq': 0.15}]

    jags_model_string = '''
        data {
                for (i in 1:length(y)) {
             z[i] <- log(y[i])
        }
    }
    
    model {
         for (i in 1:length(z)) {
             z[i] ~ dnorm(mu, tausq)
      }
      mu ~ dnorm(0, 0.0001)
      tausq ~ dgamma(0.0001, 0.0001)
      sigmasq <- 1/tausq
      expmu <- exp(mu)
    }
    
    '''
    jags_model \
        = pj.Model(code=jags_model_string,
                   init=inits,
                   data=data,
                   chains=3,
                   )

    samples_1 = jags_model.sample(iterations=1000, vars=["mu", "sigmasq", "expmu"])
    idata = az.from_pyjags(samples_1)
    fig, ax = plt.subplots(3, 2, figsize=(12, 5))
    az.plot_trace(idata,
                  var_names=["mu", "sigmasq", "expmu"], axes=ax)
    plt.tight_layout()
    plt.show()
    print(1)
    az.plot_autocorr(idata.posterior.sel(chain=0), var_names=["mu", "sigmasq", "expmu"])
    plt.show()

    idata2 = idata.sel(draw=slice(500, 1000))
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
    summary = az.summary(idata2.posterior[["mu", "sigmasq", "expmu"]], stat_funcs=func_dict)
    print(summary)

    #   Verify: Time-series SE less than 1/20 of SD
    az.plot_posterior(idata, var_names=["mu", "sigmasq", "expmu"], figsize=(12, 3))
    plt.show()
    pass


if __name__ == "__main__":
    main()

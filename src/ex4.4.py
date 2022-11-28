import arviz as az
import matplotlib.pyplot as plt
import pyjags as pj
import pandas as pd
import numpy as np


def main():
    df = pd.read_csv('../data/ex4.4data.txt', sep='  ', header=None)
    ages = np.array([8, 15, 22, 29, 36])
    data = {'Y': np.ma.masked_invalid(df.values),
            'X': np.ma.masked_invalid(ages),
            'Xbar': np.ma.masked_invalid(np.mean(ages)),
            'Omega0': np.ma.masked_invalid(np.array([[100, 0],
                                                     [0, 0.1]])),
            'mu0': np.ma.masked_invalid(np.array([0, 0])),
            'Sigma0.inv': np.ma.masked_invalid(np.array([[1e-6, 0],
                                                         [0, 1e-6]]))}

    inits = [{'tausq.y': 1, 'beta': np.array([0, 0]), 'Omega.inv': np.identity(2)},
             {'tausq.y': 100, 'beta': np.array([100, 100]), 'Omega.inv': 100 * np.identity(2)},
             {'tausq.y': 0.1, 'beta': np.array([-100, -100]), 'Omega.inv': 0.01 * np.identity(2)}]

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
        
            alpha[i,1:2] ~ dmnorm(beta, Omega.inv)
          }
        
          tausq.y ~ dgamma(0.001, 0.001)
          sigma.y <- 1 / sqrt(tausq.y)
        
          beta ~ dmnorm(mu0, Sigma0.inv)
          Omega.inv ~ dwish(2*Omega0, 2)
          Omega <- inverse(Omega.inv)
        
          rho <- Omega[1,2] / sqrt(Omega[1,1] * Omega[2,2])
        }
        '''
    jags_model \
        = pj.Model(code=jags_model_string,
                   init=inits,
                   data=data,
                   chains=3,
                   )

    samples_1 = jags_model.sample(iterations=1000, vars=["beta", "sigma.y", "rho", 'Omega'])
    samples_1['Omega'] = np.reshape(samples_1['Omega'], [4, 1000, 3])
    idata = az.from_pyjags(samples_1)

    az.plot_trace(idata, var_names=["beta", "sigma.y", "rho", "Omega"])
    plt.tight_layout()
    plt.show()

    # Autocorrelation plots for chain # 1
    az.plot_autocorr(idata.posterior.sel(chain=1), var_names=["beta", "sigma.y", "Omega", "rho"])
    plt.tight_layout()
    plt.show()

    samples_2 = jags_model.sample(iterations=11000, vars=["beta", "sigma.y", "Omega", "rho"])
    samples_2['Omega'] = np.reshape(samples_2['Omega'], [4, 11000, 3])
    idata2 = az.from_pyjags(samples_2)

    az.plot_posterior(idata2, var_names=["beta", "sigma.y", "Omega", "rho"])
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
    summary = az.summary(idata2.posterior.sel(draw=slice(4000, 11000))[["beta", "sigma.y", "Omega", "rho"]],
                         stat_funcs=func_dict)
    print(summary)

    fig, ax = plt.subplots()
    ax.scatter(idata2.posterior.sel(draw=slice(4000, 11000)).beta.sel(beta_dim_0=0),
               idata2.posterior.sel(draw=slice(4000, 11000)).beta.sel(beta_dim_0=1), s=0.1, fc='k')
    ax.set_xlabel('beta1')
    ax.set_ylabel('beta2')
    plt.show()
    print(1)


if __name__ == "__main__":
    main()

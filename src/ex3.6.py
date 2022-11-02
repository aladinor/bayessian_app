import arviz as az
import matplotlib.pyplot as plt
import pyjags as pj


def main():
    data = {'y': 12, 'n': '70', 'alpha': 0.5, 'beta': 0.5}
    inits = [{'theta': 0.1}, {'theta': 0.5}, {'theta': 0.9}]

    jags_model_string = '''
    model {
          y ~ dbin(theta, n)
          theta ~ dbeta(alpha, beta)
    }'''

    jags_model \
        = pj.Model(code=jags_model_string,
                   init=inits,
                   data=data,
                   chains=3,
                   adapt=0,
                   )
    samples_1 = jags_model.sample(iterations=1000)

    idata = az.from_pyjags(samples_1)
    az.plot_trace(idata,
                  var_names=['theta'])
    plt.show()
    az.plot_autocorr(idata, var_names=['theta'])
    plt.show()
    print(1)
    pass


if __name__ == "__main__":
    main()

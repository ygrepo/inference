import pyro

import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
from pyro.distributions import Gamma, Poisson, Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

import logging
from src.practice.utils import  Pmf, make_gaussian_pmf, eval_poisson_pmf

USE_SUMMARY_DATA = True

class Hockey(Pmf):
    """Represents hypotheses about the scoring rate for a team."""

    def __init__(self, name=''):
        """Initializes the Hockey object.

        name: string
        """
        if USE_SUMMARY_DATA:
            # prior based on each team's average goals scored
            mu = 2.8
            sigma = 0.3
        else:
            # prior based on each pair-wise match-up
            mu = 2.8
            sigma = 0.85

        pmf = make_gaussian_pmf(mu, sigma, 4)
        Pmf.__init__(self, pmf, name=name)

    def likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        Evaluates the Poisson PMF for lambda and k.

        hypo: goal scoring rate in goals per game
        data: goals scored in one period
        """
        lam = hypo
        k = data
        like = eval_poisson_pmf(k, lam)
        return like

def train_model(data, n_steps, pmf):

    def model(data):
        mu = 2.8
        num_sigmas = 4
        sigma = 0.3
        low = mu - num_sigmas * sigma
        high = mu + num_sigmas * sigma

        f = pyro.sample("latent", dist.Uniform(low, high))
        print(f)
        # sample f from the prior
        # Probabilities are generated by the pmf
        for i in range(len(data)):
            pyro.sample("obs_{}".format(i), Poisson(f), obs=data[i])


    def guide(data):
        lam = pyro.param("lam", torch.tensor(2.0), constraint=constraints.positive)
        # alpha_q = pyro.param("alpha_q", torch.tensor(2.0))
        # beta_q = pyro.param("beta_q", torch.tensor(1.0))
        pyro.sample("latent", dist.Poisson(torch.tensor(2.0)))

    adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
    optimizer = ClippedAdam(adam_params)

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    for step in range(n_steps):
        loss = svi.step(data)
        if step % 100 == 0:
            logging.info(".")
            logging.info("Elbo loss: {}".format(loss))

    # grab the learned variational parameters
    lam = pyro.param("lam").item()
    print(lam)
    # a_q = pyro.param("alpha_q").item()
    # b_q = pyro.param("beta_q").item()
    # print(a_q, b_q)
    posterior = Poisson(lam)
    logging.info("Sampling:{}".format(posterior.sample()))


logging.basicConfig(format='%(message)s', level=logging.INFO)

# enable validation (e.g. validate parameters of distributions)
# assert pyro.__version__.startswith('0.3.1')
pyro.enable_validation(True)

# clear the param store in case we're in a REPL
pyro.clear_param_store()

n_steps = 500
print("n_steps={}".format(n_steps))

pyro.set_rng_seed(1)
pmf1 = Hockey("bruins")
data = [0.0, 2.0, 8.0, 4.0]
pmf1.update_set(data)
tdata = []
for v in data:
    tdata.append(torch.tensor(v))
train_model(tdata, n_steps, pmf1)

#pmf2 = Hockey("canucks")


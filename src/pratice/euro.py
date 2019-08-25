import logging
import math
import os

import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from src.pratice.utils import Pmf

logging.basicConfig(format='%(message)s', level=logging.INFO)

# enable validation (e.g. validate parameters of distributions)
# assert pyro.__version__.startswith('0.3.1')
pyro.enable_validation(True)

# clear the param store in case we're in a REPL
pyro.clear_param_store()

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
n_steps = 2 if smoke_test else 2000
n_steps = 500
print("n_steps={}".format(n_steps))

pyro.set_rng_seed(1)

data = []
for _ in range(140):
    data.append(torch.tensor(1.0))
for _ in range(110):
    data.append(torch.tensor(0.0))


class Euro(Pmf):
    """Represents hypotheses about the probability of heads."""

    def likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: integer value of x, the probability of heads (0-100)
        data: string 'H' or 'T'
        """
        x = hypo / 100.0
        if data == "H":
            return x
        else:
            return 1 - x


def uniform_prior():
    pmf = Euro(range(0, 101))
    return pmf


def triangle_prior():
    pmf = Euro()
    for x in range(0, 51):
        pmf.set(x, x)
    for x in range(51, 101):
        pmf.set(x, 100 - x)
    return pmf


def update(pmf, heads=140, tails=110):
    """Updates the pmf with the given number of heads and tails.

    pmf: Pmf object
    heads: int
    tails: int
    """
    dataset = "H" * heads + "T" * tails

    for data in dataset:
        pmf.update(data)


def train_model(pmf):
    def model(data):
        # sample f from the prior
        # Probabilities are generated by the pmf
        f = pyro.sample("latent_fairness", pmf)
        f2 = dist.Bernoulli(f)
        for i in range(len(data)):
            s = pyro.sample("obs_{}".format(i), f2, obs=data[i])

    def guide(data):
        alpha_q = pyro.param("alpha_q", torch.tensor(15.0),
                             constraint=constraints.positive)
        beta_q = pyro.param("beta_q", torch.tensor(15.0),
                            constraint=constraints.positive)
        # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
        pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))

    adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
    optimizer = ClippedAdam(adam_params)

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    for step in range(n_steps):
        loss = svi.step(data)
        if step % 100 == 0:
            logging.info(".")
            logging.info("Elbo loss: {}".format(loss))

    # grab the learned variational parameters
    a_q = pyro.param("alpha_q").item()
    b_q = pyro.param("beta_q").item()

    inferred_mean = a_q / (a_q + b_q)
    # compute inferred standard deviation
    factor = b_q / (a_q * (1.0 + a_q + b_q))
    inferred_std = inferred_mean * math.sqrt(factor)
    print("\nbased on the data and our prior belief, the fairness " +
          "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))

    beta_posterior = torch.distributions.beta.Beta(a_q, b_q)
    posterior = torch.distributions.bernoulli.Bernoulli(beta_posterior.sample())
    logging.info("Sampling:{}".format(posterior.sample()))


pmf = uniform_prior()
update(pmf)
train_model(pmf)
pmf = triangle_prior()
update(pmf)
train_model(pmf)
alpha0 = torch.tensor(10.0)
beta0 = torch.tensor(10.0)
# sample f from the beta prior
train_model(dist.Beta(alpha0, beta0))

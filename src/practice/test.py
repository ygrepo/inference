
import torch
import math
import os

import numpy as np
from scipy.stats import uniform

import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
from torch.distributions import Uniform, Poisson
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

import pyro.poutine as poutine


def model(x):
    s = pyro.param("s", torch.tensor(0.5))
    z = pyro.sample("z", dist.Normal(x, s))
    return z ** 2
p = pyro.sample("latent", dist.Uniform(2, 5))
print(p)
z = pyro.sample("z", dist.Poisson(p))
print(z)
trace = pyro.poutine.trace(model).get_trace(0.0)
print(trace)
print(trace.nodes)
print(trace.nodes["_RETURN"]["value"])

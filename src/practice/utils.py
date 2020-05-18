import logging
import math

import numpy as np
from scipy import stats


def get_logger(log_file):
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=log_file, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    def log(s):
        logging.info(s)

    return log


class UnimplementedMethodException(Exception):
    """Exception if someone calls a method that should be overridden."""


class Pmf(object):

    def __init__(self, values=None, name=""):
        self.name = name
        self.d = {}
        self.log = False
        if values is None:
            return

        init_methods = [
            self.init_pmf,
            self.init_mapping,
            self.init_sequence,
            self.init_failure,
        ]

        for method in init_methods:
            try:
                method(values)
                break
            except AttributeError:
                continue

        if len(self) > 0:
            self.normalize()

    def get_items(self):
        """Gets an unsorted sequence of (value, freq/prob) pairs."""
        return self.d.items()

    def init_mapping(self, values):
        """Initializes with a map from value to probability.

        values: map from value to probability
        """
        for value, prob in values.iteritems():
            self.set(value, prob)

    def init_pmf(self, values):
        """Initializes with a Pmf.

        values: Pmf object
        """
        for value, prob in values.items():
            self.set(value, prob)

    def init_sequence(self, values):
        """Initializes with a sequence of equally-likely values.

        values: sequence of values
        """
        for value in values:
            self.set(value, 1)


    def init_failure(self, values):
        """Raises an error."""
        raise ValueError('None of the initialization methods worked.')

    def sample(self, n):
        """Generates a random sample from this distribution.

        Args:
            n: int length of the sample
        """
        raise UnimplementedMethodException()

    def likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: some representation of the hypothesis
        data: some representation of the data
        """
        raise UnimplementedMethodException()

    def set(self, x, y=0):
        """Sets the freq/prob associated with the value x.

        Args:
            x: number value
            y: number freq or prob
        """
        self.d[x] = y

    def remove(self, x):
        """Removes a value.

        Throws an exception if the value is not there.

        Args:
            x: value to remove
        """
        del self.d[x]

    def values(self):
        """Gets an unsorted sequence of values.

        Note: one source of confusion is that the keys of this
        dictionary are the values of the Hist/Pmf, and the
        values of the dictionary are frequencies/probabilities.
        """
        return self.d.keys()

    def items(self):
        """Gets an unsorted sequence of (value, freq/prob) pairs."""
        return self.d.items()

    def mult(self, x, factor):
        """Scales the freq/prob associated with the value x.

        Args:
            x: number value
            factor: how much to multiply by
        """
        self.d[x] = self.d.get(x, 0) * factor

    def total(self):
        """Returns the total of the frequencies/probabilities in the map."""
        total = sum(self.d.values())
        return total

    def max_like(self):
        """Returns the largest frequency/probability in the map."""
        return max(self.d.values())

    def normalize(self, fraction=1.0):
        """Normalizes this PMF so the sum of all probs is fraction.

        Args:
            fraction: what the total should be after normalization

        Returns: the total probability before normalizing
        """
        if self.log:
            raise ValueError("Pmf is under a log transform")

        total = self.total()
        if total == 0.0:
            raise ValueError('total probability is zero.')
            logging.warning('Normalize: total probability is zero.')
            return total

        factor = float(fraction) / total
        for x in self.d:
            self.d[x] *= factor

        return total

    def update_set(self, dataset):
        """Updates each hypothesis based on the dataset.

        This is more efficient than calling Update repeatedly because
        it waits until the end to Normalize.

        Modifies the suite directly; if you want to keep the original, make
        a copy.

        dataset: a sequence of data

        returns: the normalizing constant
        """
        for data in dataset:
            for hypo in self.values():
                like = self.likelihood(data, hypo)
                self.mult(hypo, like)
        return self.normalize()

    def update(self, data):
        """Updates each hypothesis based on the data.

        data: any representation of the data

        returns: the normalizing constant
        """
        for hypo in self.values():
            like = self.likelihood(data, hypo)
            self.mult(hypo, like)

        return self.normalize()

    def prob(self, x, default=0):
        """Gets the probability associated with the value x.

        Args:
            x: number value
            default: value to return if the key is not there

        Returns:
            float probability
        """
        return self.d.get(x, default)

    def probs(self, xs):
        """Gets probabilities for a sequence of values."""
        return [self.prob(x) for x in xs]

    def print(self):
        """Prints the hypotheses and their probabilities."""
        for hypo, prob in sorted(self.items()):
            print(hypo, prob)

    def log_prob_normalize(self, m=None):
        """Log transforms the probabilities.

        Removes values with probability 0.

        Normalizes so that the largest logprob is 0.
        """
        if self.log:
            raise ValueError("Pmf/Hist already under a log transform")
        self.log = True

        if m is None:
            m = self.max_like()

        new_d = {}
        for x, p in self.d.items():
            if p:
                new_d[x] = math.log(p / m)
        self.d = new_d

    def log_prob(self, x):
        ar = np.array(self.prob(x))
        return ar

    def __len__(self):
        return len(self.d)

    def __iter__(self):
        return iter(self.d)

    def iterkeys(self):
        return iter(self.d)

    def __contains__(self, value):
        return value in self.d


def eval_gaussian_pdf(x, mu, sigma):
    """Computes the unnormalized PDF of the normal distribution.

    x: value
    mu: mean
    sigma: standard deviation

    returns: float probability density
    """
    return stats.norm.pdf(x, mu, sigma)


def make_gaussian_pmf(mu, sigma, num_sigmas, n=201):
    """Makes a PMF discrete approx to a Gaussian distribution.

    mu: float mean
    sigma: float standard deviation
    num_sigmas: how many sigmas to extend in each direction
    n: number of values in the Pmf

    returns: normalized Pmf
    """
    pmf = Pmf()
    low = mu - num_sigmas * sigma
    high = mu + num_sigmas * sigma

    for x in np.linspace(low, high, n):
        p = eval_gaussian_pdf(x, mu, sigma)
        pmf.set(x, p)
    pmf.normalize()
    return pmf

def eval_poisson_pmf(k, lam):
    """Computes the Poisson PMF.

    k: number of events
    lam: parameter lambda in events per unit time

    returns: float probability
    """
    return stats.poisson.pmf(k, lam)

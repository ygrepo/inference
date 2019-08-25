import logging
import math

import numpy as np


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
        self.init_sequence(values)
        self.normalize()

    def get_items(self):
        """Gets an unsorted sequence of (value, freq/prob) pairs."""
        return self.d.items()

    def init_sequence(self, values):
        """Initializes with a sequence of equally-likely values.

        values: sequence of values
        """
        for value in values:
            self.set(value, 1)

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
        return np.array(self.prob(x))

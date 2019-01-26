from __future__ import print_function, unicode_literals
__docformat__ = "epytext en"

try:
    import numpy
except ImportError:
    pass

import tempfile
import os
from collections import defaultdict
from nltk import compat
# from nltk.data import gzip_open_unicde
from nltk.util import OrderedDict
from nltk.probability import DictionaryProbDist
from nltk.classify.api import ClassifierI
from nltk.classify.util import CutoffChecker, accuracy, log_likelihood
from nltk.classify.megam import (call_megam, write_megam_file, parse_megam_weights)
from nltk.classify.tadm import call_tadm, write_tadm_file, parse_tadm_weights

# -*- coding: utf-8 -*-
# MLE : Maximum Likelihood Estimation
import nltk
from nltk.probability import *

class MLEProbDist(ProbDistI):
    def __init__(self, freqdist, bins=None):
        self._freqdist = freqdist

    def freqdist(self):
        return self._freqdist

    def prob(self, sample):
        return self._freqdist.freq(sample)

    def max(self):
        return self._freqdist.max()

    def samples(self):
        return self._freqdist.keys()

    def __repr__(self):
        return "MLEProbDist based on {} sampled".format(self._freqdist.N())

class LidstoneProbDist(ProbDistI):
    SUM_TO_ONE = False
    def __init__(self, freqdist, gamma, bins=None):
        if bins ==0 or (bins is None and freqdist.N()==0):
            name = self.__class__.__name__[:-8]
            raise ValueError("A {} probability distribution".format(name+"must have at least one bin"))

        if (bins is not None) and (bins < freqdist.B()):
            name = self.__class__.__name__[:-8]
            raise ValueError("\nThe number of bins in a {} distribution".format(name+"{} must be greater than or equal to\n".format(bins)+"the number of bins in the FreqDist used to create it ({}).".format(freqdist.B())))

            self._freqdist = freqdist
            self._gamma = float(gamma)
            self._N = self._freqdist.N()

        if bins is None:
            bins = freqdist.B()
        self._bins = bins

        self._divisor = self._N + bins * gamma
        if self._divisor == 0.0:
            self._gamma = 0
            self._divisor = 1

    def freqdist(self):
        return self._freqdist

    def prob(self, sample):
        c = self._freqdist[sample]
        return (c+self._gamma)/self._divisor

    def max(self):
        return self._freqdist.max()

    def samples(self):
        return self._freqdist.keys()

    def discount(self):
        gb = self._gamma * self._bins
        return gb/(self._N + gb)

    def __repr__(self):
        return "<LidstoneProbDist based on {} samples>".format(self._freqdist.N())

class LaplaceProbDist(LidstoneProbDist):
    def __init__(self, freqdist, bins=None):
        LidstoneProbDist.__init__(self, freqdist, 1, bins)

    def __repr__(selfself):
        return "<LaplaceProbDist based on {} samples>".format(self._freqdist.N())

class ELEProbDist(LidstoneProbDist):
    def __init__(self, freqdist, bins=None):
        LidstoneProbDist.__init__(self, freqdist, 0.5, bins)

    def __repr__(self):
        return "<ELEProbDist based on {} samples>".format(self._freqdist.N())

class WittenBellProbDist(ProbDistI):
    def __init__(self, freqdist, bins=None):
        assert bins is None or bins>=freqdist.B(), "bins parameter must not be less than {}=freqdist.B()".format(freqdist.B())

        if bins is None:
            bins = freqdist.B()

        self._freqdist = freqdist
        self._T = self._freqdist.B()
        self._Z = bins - self._freqdist.B()
        self._N = self_freqdist.N()

        if self._N == 0:
            self._P0 = 1.0 / self._Z
        else:
            self._P0 = self._T / float(self._Z * (self._N + self._T))

    def prob(self, sample):
        c = self._freqdist[sample]
        return (c/float(self._N + self._T) if c!=0 else self._P0)

    def max(self):
        return self._freqdist.max()

    def samples(self):
        return self._freqdist.keys()

    def freqdist(self):
        return self._freqdist

    def discount(self):
        raise NotImplementedError()

    def __repr__(self):
        return "<WittenBellProbDist based on {} samples>".format(self._freqdist.N())


if __name__ == "__main__":
    print(train_and_test(mle))
    print(train_and_test(LaplaceProbDist))
    print(train_and_test(ELEProbDist))

    def lidstone(gamma):
        return lambda fd, bins: LidstoneProbDist(fd, gamma, bins)

    print(train_and_test(lidstone(0.1)))
    print(train_and_test(lidstone(0.5)))
    print(train_and_test(lidstone(1.0)))


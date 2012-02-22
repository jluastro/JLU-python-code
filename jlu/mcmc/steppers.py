""" Class to hold the model variables and do some fitting"""

import numpy as np
import pymc

class HitAndRun(pymc.Gibbs):
    def __init__(self, stochastic, proposal_sd=None, verbose=None):
        pymc.Metropolis.__init__(self, stochastic, proposal_sd=proposal_sd,
                                 verbose=verbose, tally=False)
        self.proposal_tau = self.proposal_sd**-2.
        
    def step(self):
        x0 = np.copy(self.stochastic.value)
        dx = pymc.rnormal(np.zeros(np.shape(x0)), self.proposal_tau)

        logp = [self.logp_plus_loglike]
        x_prime = [x0]

        for direction in [-1, 1]:
            for i in xrange(25):
                delta = direction*np.exp(.1*i)*dx
                try:
                    self.stochastic.value = x0 + delta
                    logp.append(self.logp_plus_loglike)
                    x_prime.append(x0 + delta)
                except pymc.ZeroProbability:
                    self.stochastic.value = x0
        
        i = pymc.rcategorical(np.exp(np.array(logp) - pymc.flib.logsum(logp)))
        self.stochastic.value = x_prime[i]

        if i == 0:
            self.rejected += 1
            if self.verbose > 2:
                print self._id + ' rejecting'
        else:
            self.accepted += 1
            if self.verbose > 2:
                print self._id + ' accepting'

class HRAM(pymc.Gibbs):
    def __init__(self, stochastic, proposal_sd=None, verbose=None):
        pymc.Metropolis.__init__(self, stochastic, proposal_sd=proposal_sd,
                                 verbose=verbose, tally=False)
        self.proposal_tau = self.proposal_sd**-2.
        self.n = 0
        self.N = 11
        self.value = pymc.rnormal(self.stochastic.value, self.proposal_tau, size=tuple([self.N] + list(self.stochastic.value.shape)))

    def step(self):
        x0 = self.value[self.n]
        u = pymc.rnormal(np.zeros(self.N), 1.)
        dx = np.dot(u, self.value)

        self.stochastic.value = x0
        logp = [self.logp_plus_loglike]
        x_prime = [x0]

        for direction in [-1, 1]:
            for i in xrange(25):
                delta = direction*np.exp(.1*i)*dx
                try:
                    self.stochastic.value = x0 + delta
                    logp.append(self.logp_plus_loglike)
                    x_prime.append(x0 + delta)
                except pymc.ZeroProbability:
                    self.stochastic.value = x0
        
        i = pymc.rcategorical(np.exp(np.array(logp) - pymc.flib.logsum(logp)))
        self.value[self.n] = x_prime[i]
        self.stochastic.value = x_prime[i]

        if i == 0:
            self.rejected += 1
            if self.verbose > 2:
                print self._id + ' rejecting'
        else:
            self.accepted += 1
            if self.verbose > 2:
                print self._id + ' accepting'

        self.n += 1
        if self.n == self.N:
            self.n = 0

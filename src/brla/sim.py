"""Bayesian RLA simulations
Based on Michigan pilot RLA values and code in BayesianWithPriorFiner.m
"""

import math
import numpy as np
from scipy import stats


def main():

    N = 35295
    N = 5
    m = 10000
    m = 3
    n = list(range(1, m+1))

    NumberErrorValues = 1
    BayesianError = (0.05, )

    HalfN = math.floor(N / 2)
    BaseLine = 1 / (N + 1)

    BayesianPrior = np.concatenate((
        np.full((HalfN,), 0.0),
        np.full((1,), 0.5),
        np.full((N-HalfN,), 0.5 / (N - HalfN)) ))

    print(BayesianPrior)

    kprev = np.zeros((NumberErrorValues, m + 1), np.int8)
    kmin = np.zeros((NumberErrorValues, m), np.int8)
    Error = np.zeros((NumberErrorValues, m))
    Nrange = np.array(range(N+1))

    for i in range(NumberErrorValues):
        kprev[i, 1] = 0
        for j in range(m):
            k = 0 # TODO: Check this
            ThisError = 0.0 # TODO: Check this
            hyp = stats.hypergeom(N, Nrange, n[j])
            for k in range(kprev[i, j], n[j]):
                # first time: hygepdf(k,N,0:1:N,n(j)) [:15] == 0.00000 0.00003 0.00006 0.00008 0.00011 0.00014 0.00017 0.00020 0.00023 0.00025 0.00028 0.00031 0.00034 0.00037
                hyppmf = hyp.pmf(k)
                BayesPosteriorDist = BayesianPrior * hyppmf  # . * hygepdf(k, N, 0:1:N, n[j])
                BayesPosteriorDist = BayesPosteriorDist / BayesPosteriorDist.sum()
                ThisError = sum(BayesPosteriorDist[:HalfN])
                print("i: %d, j: %d, k: %d, e: %f, hyppmf: %s BPD: %s" % (i, j, k, ThisError, hyppmf, BayesPosteriorDist))
                if ThisError <= BayesianError[i]:
                    break

            kmin[i, j] = k
            kprev[i, j + 1] = k
            Error[i, j] = ThisError

        for i in range(NumberErrorValues):
            for j in range(m):
                if kmin[i, j] == n[j]:
                    kmin[i, j] = 0
                    Error[i, j] = 0
    print("kprev: %s\nkmin: %s\nerror: %s" % (kprev, kmin, Error))


if __name__ == "__main__":
    main()

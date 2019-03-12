"""Bayesian RLA simulations
Based on Michigan pilot RLA values and code in BayesianWithPriorFiner.m
"""

import math
import numpy as np
import logging
from scipy import stats


def main():

    # logging.basicConfig(level=logging.DEBUG)
    N = 35295
    N = 100000
    #N = 5
    #m = 3
    # n = list(range(1, 10000 + 1))
    n = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
    m = len(n)

    BayesianError = (0.05, 0.005, 0.1)
    NumberErrorValues = len(BayesianError)

    HalfN = math.floor(N / 2)
    BaseLine = 1 / (N + 1)

    BayesianPrior = np.concatenate((
        np.full((HalfN,), 0.0),
        np.full((1,), 0.5),
        np.full((N-HalfN,), 0.5 / (N - HalfN)) ))

    # print(BayesianPrior)

    kprev = np.zeros((NumberErrorValues, m + 1), np.int)
    kmin = np.zeros((NumberErrorValues, m), np.int)
    Error = np.zeros((NumberErrorValues, m))
    Nrange = np.array(range(N+1))

    old_err_state = np.seterr(divide='raise')

    # try: 
    for i in range(NumberErrorValues):
        kprev[i, 0] = n[0] // 2
        for j in range(m):
            hyp = stats.hypergeom(N, Nrange, n[j])
            for k in range(n[j] // 2, n[j]+1):  # or max of that and kprev? And test requirement that n[j] is increasing?
                # first time: hygepdf(k,N,0:1:N,n(j)) [:15] == 0.00000 0.00003 0.00006 0.00008 0.00011 0.00014 0.00017 0.00020 0.00023 0.00025 0.00028 0.00031 0.00034 0.00037
                hyppmf = hyp.pmf(k)
                BayesPosteriorDist = BayesianPrior * hyppmf  # . * hygepdf(k, N, 0:1:N, n[j])
                BayesPosteriorDist = BayesPosteriorDist / BayesPosteriorDist.sum()
                ThisError = sum(BayesPosteriorDist[:HalfN+1])
                logging.debug("i: %d, j: %d, k: %d, e: %f, hyppmf: %s BPD: %s" % (i, j, k, ThisError, hyppmf, BayesPosteriorDist))
                if ThisError <= BayesianError[i]:
                    break

            kmin[i, j] = k
            kprev[i, j + 1] = k
            Error[i, j] = ThisError
            print("limit: %.3f, size: %5d, k: %5d, k/n[j]: %.4f, e: %f" % (BayesianError[i], n[j], k, k / n[j], ThisError))

        for i in range(NumberErrorValues):
            for j in range(m):
                if kmin[i, j] == n[j]:
                    kmin[i, j] = 0
                    Error[i, j] = 0

    #except FloatingPointError as e:
    #    print(e)

    # print("n: %s\nNrange: %s\nkprev: %s\nkmin: %s\nerror: %s" % (n, Nrange, kprev, kmin, Error))
    print("kmin: %s\nerror: %s" % (kmin, Error))

if __name__ == "__main__":
    main()

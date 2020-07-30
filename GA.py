# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
"""

import numpy as np
import matplotlib.pyplot as plt


def GA(n, max_evals, decodefct, selectfct, fitnessfct, max_attainable=np.inf, seed=None):
    eval_cntr = 0
    history = []
    #
    # GA params
    mu = 1000
    pc = 0.37
    pm = 4 / n
    #    kXO = 1 # 1-point Xover
    local_state = np.random.RandomState(seed)
    Genome = local_state.randint(2, size=(n, mu))
    Phenotype = []
    for k in range(mu):
        Phenotype.append(decodefct(Genome[:, [k]]))
    # print(Phenotype[k],type(Phenotype[k]))
    #    print('===')
    fitness = fitnessfct(Phenotype)
    eval_cntr += mu
    fcurr_best = fmax = np.max(fitness)
    xmax = Genome[:, [np.argmin(fitness)]]
    history.append(fmax)
    while (eval_cntr < max_evals):
        #       Generate offspring population (recombination, mutation)
        newGenome = np.empty([n, mu], dtype=int)
        #        1. sexual selection + 1-point recombination
        for i in range(int(mu / 2)):
            p1 = selectfct(Genome, fitness, local_state)
            p2 = selectfct(Genome, fitness, local_state)
            if local_state.uniform() < pc:  # recombination
                idx = local_state.randint(n, dtype=int)
                Xnew1 = np.concatenate((p1[:idx], p2[idx:]))
                Xnew2 = np.concatenate((p2[:idx], p1[idx:]))
            else:  # no recombination; two parents are copied as are
                Xnew1 = p1
                Xnew2 = p2
            # 2. mutation
            mut1_bits = local_state.uniform(size=(n, 1)) < pm
            mut2_bits = local_state.uniform(size=(n, 1)) < pm
            Xnew1[mut1_bits] = 1 - Xnew1[mut1_bits]
            Xnew2[mut2_bits] = 1 - Xnew2[mut2_bits]
            #
            newGenome[:, [2 * i - 1]] = Xnew1
            newGenome[:, [2 * i]] = Xnew2
        # The best individual of the parental population is kept
        newGenome[:, [mu - 1]] = Genome[:, [np.argmax(fitness)]]
        Genome = newGenome
        Phenotype.clear()
        for k in range(mu):
            Phenotype.append(decodefct(Genome[:, [k]]))
        fitness = fitnessfct(Phenotype)
        eval_cntr += mu
        fcurr_best = np.max(fitness)
        if fmax < fcurr_best:
            fmax = fcurr_best
            xmax = Genome[:, [np.argmax(fitness)]]
        history.append(fcurr_best)
        if np.mod(eval_cntr, int(max_evals / 10)) == 0:
            print(eval_cntr, " evals: fmax=", fmax)
        if fmax == max_attainable:
            print(eval_cntr, " evals: fmax=", fmax, "; done!")
            break
    return xmax, fmax, history


#
def ONEMAX(PhenoList):
    '''The counting-ones problem implemented for evaluating the entire population.
    '''
    mu = len(PhenoList)
    fitness = np.empty([mu], dtype=int)
    for i in range(mu):
        fitness[i] = (int)(np.sum(PhenoList[i]))
    return fitness


#
# def TeleCom(PhenoList):
#     """ The correlation function, assumes a numpy vector {-1,+1} as input """
#     mu = len(PhenoList)
#     fitness = np.empty([mu])
#     for i in range(mu):
#         fitness[i] = objFunctions.SwedishPump(PhenoList[i].ravel())
#     return fitness


#
def no_decoding(a):
    ''' The identity function when no decoding is needed, e.g., search over binary landscapes. '''
    return a


#
def decoding_ones(a):
    z = a < 1
    a[z] = -1
    return a


#
def select_proportional(Genome, fitness, rand_state):
    ''' RWS: Select one individual out of a population Genome with fitness values fitness using proportional selection.'''
    cumsum_f = np.cumsum(fitness)
    r = sum(fitness) * rand_state.uniform()
    idx = np.ravel(np.where(r < cumsum_f))[0]
    return Genome[:, [idx]]


if __name__ == "__main__":
    n = 64
    evals = 10 ** 8
    Nruns = 10
    fbest = []
    xbest = []
    for i in range(Nruns):
        xmax, fmax, history = GA(n, evals, decoding_ones, select_proportional, ONEMAX, n, i + 37)
        #        xmax,fmax,history = GA(n,evals,no_decoding,select_proportional,ONEMAX,n,i+37)
        plt.semilogy(np.array(history))
        plt.show()
        print(i, ": maximal ONEMAX found is ", fmax, " at location ", xmax.T)
        fbest.append(fmax)
        xbest.append(xmax)
    print("====\n Best ever: ", max(fbest), "x*=", xbest[fbest.index(max(fbest))].T)
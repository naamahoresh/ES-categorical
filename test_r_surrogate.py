import math
import numpy as np
import os
import errno
import random
import heapq
import re
from multiprocessing import Process, Lock
import psutil
from scipy import stats
import warnings

import sys

sys.path.insert(0, '/home/naamah/Documents/CatES/')

from NKLandscape_class import NKlandscape
from EdgeCrossOver import ERO
from time import sleep

from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# try:
#     import cPickle as pickle
# except:
import pickle as pickle

np.set_printoptions(threshold=np.nan)
import surrogate_withR as surrogate
from datetime import datetime


class CatES:
    def __init__(self, fitnessfun, DEVector, typeExp, numPar, popSize, seed=0, max_attainable=np.inf, niter=100,
                 opt='max', repeated_Number=False, sourrogate="KRIGING", saveNum=0, f_function_counter=600,
                 withSurr=True, problem="QAP", pmut=0.4, lowest_pmut=0.005, lowest_pmut_defult=0.01, pCrossOver=0.5, Tournament_size=10,
                 num_elitism=2, XInit=None,RF_parm=None,svm_parm=None,knn_parm=None):
        self.N = numPar  # number of parameters
        self.popSize = popSize  # number of sample for each generation
        self.niter = niter  # number iterations
        self.typeExp = typeExp;  # lab expirance or simulation. 0 = lab. 1 = simulation
        self.fitness = fitnessfun
        # self.local_state = np.random.RandomState(seed)
        self.fileLocSave_general = "/home/naamah/Documents/CatES/result_All/" + problem + "/" + sourrogate + "_" + str(
            saveNum)
        self.fileLocSave_exp_num = self.fileLocSave_general + "/exp_num_" + str(seed + 1) + "/"
        self.fileLocSave_bythread = self.fileLocSave_exp_num
        self.max_attainable = max_attainable
        self.opt = opt
        self.Farchive = np.empty(0)
        self.Xarchive = np.empty(0)
        self.Sarchive = np.empty(0)
        self.repeated_Number = repeated_Number
        self.withArchive = True
        self.sourrogateTyp = sourrogate
        self.f_function_counter = f_function_counter
        self.withSurr = withSurr
        self.problem = problem
        self.crate_folder(self.fileLocSave_exp_num)
        self.seed = seed

        self.surr_Farchive = []
        self.surr_Xarchive = []
        self.surrTest = []

        self.pCrossOver = pCrossOver
        self.Tournament_size = Tournament_size
        self.num_elitism = num_elitism
        self.lowest_pmut = lowest_pmut
        self.lowest_pmut_defult = lowest_pmut_defult
        self.pmut = pmut

        self.XInit = XInit
        # inti DE - number of levels for each parameter. 2 options:
        # 1. all the parameters have same lavel
        # 2. each parapmeter have different lavel
        if (DEVector[0] == 0):
            self.DE = np.full((1, self.N), DEVector[1]).squeeze()
        else:
            self.DE = np.array(DEVector[1:self.N + 1])
        self.RF_parm=RF_parm
        self.svm_parm=svm_parm
        self.knn_parm=knn_parm
    def set_exp(self):
        iter = 0
        if (not iter):
            Xnew = self.init_Exp()
        else:
            Xnew = self.step_Exp(iter)

    # main loop of the algorotem
    def test_exp(self, numThread):
        self.fileLocSave_bythread = self.fileLocSave_bythread + "threadNum_" + str(numThread + 1) + "/"
        self.crate_folder(self.fileLocSave_bythread)
        fopt = []
        Xopt = []
        fopt_byFitnessF = []

        X = self.init_Exp()
        F = self.eval_inExp(X, True)
        self.withArchive = True
        self.write_Exp('exp_Farchive_0.dat', F, 0, withArchive=self.withArchive)

        # experiment step: turnment, mutation and crossover.
        # Then, evaluate the fitness and write it to file
        niter_counter = self.niter
        for i in range(1, self.niter + 1):  # num iterations
            if self.f_function_counter > 0:  # num calls to the F function
                Xnew = self.step_Exp(i)

                if self.withSurr:  # exp with surrogate
                    if (i < 20 or i % 9 == 0 or i % 10 == 0):
                        F = self.eval_inExp(Xnew[0:len(Xnew) - 1, :], True)
                        self.withArchive = True
                        Ftest = F
                    elif (i == 21 and self.sourrogateTyp == "KRIGING"):
                        F = surrogate.surrogateKriging(self.surr_Xarchive[:, 0:len(self.surr_Farchive) + 1],
                                                       self.surr_Farchive,
                                                       Xnew[0:self.N, :], self.fileLocSave_bythread,
                                                       self.fileLocSave_general, self.withArchive, True,
                                                       self.problem)  # call to the soraggate model
                        Ftest = self.eval_inExp(Xnew[0:len(Xnew) - 1, :], False)
                        self.surrTest.append(np.linalg.norm(Ftest - F))
                        self.withArchive = False

                    else:
                        if self.sourrogateTyp == "RBFN":
                            F = surrogate.surrogateRBFN(self.surr_Xarchive[:, 0:len(self.surr_Farchive)],
                                                        self.surr_Farchive,
                                                        Xnew[0:self.N, :], self.fileLocSave_bythread,
                                                        self.fileLocSave_general, self.withArchive, False,
                                                        self.problem)  # call to the soraggate model
                        elif self.sourrogateTyp == "KRIGING":
                            F = surrogate.surrogateKriging(self.surr_Xarchive, self.surr_Farchive,
                                                           Xnew[0:self.N, :], self.fileLocSave_bythread,
                                                           self.fileLocSave_general, self.withArchive, False,
                                                           self.problem)  # call to the soraggate model

                        elif self.sourrogateTyp == "RF":
                            F = surrogate.surrogateRF(self.surr_Xarchive, self.surr_Farchive,
                                                      Xnew[0:self.N, :], self.fileLocSave_bythread,
                                                      self.fileLocSave_general, self.withArchive, False,
                                                      self.problem,self.RF_parm)

                        elif self.sourrogateTyp == "KNN":
                            F = surrogate.surrogateKNN(self.surr_Xarchive, self.surr_Farchive,
                                                       Xnew[0:self.N, :], self.fileLocSave_bythread,
                                                       self.fileLocSave_general, self.withArchive, False,
                                                       self.problem, self.knn_parm)

                        else:
                            F = surrogate.surrogateSVM(self.surr_Xarchive, self.surr_Farchive,
                                                       Xnew[0:self.N, :], self.fileLocSave_bythread,
                                                       self.fileLocSave_general, self.withArchive, False,
                                                       self.problem,self.svm_parm)

                        # check the diff between the surrogate and the real fitness function
                        Ftest = self.eval_inExp(Xnew[0:len(Xnew) - 1, :], False)
                        self.surrTest.append(np.linalg.norm(Ftest - F))
                        self.withArchive = False

                else:  # exp without surrogate
                    F = self.eval_inExp(Xnew[0:len(Xnew) - 1, :], True)

                self.write_Exp('exp_Xarchive_' + str(i) + '.dat', Xnew[0:self.N, :], self.niter,
                               withArchive=self.withArchive)
                self.write_Exp(str("exp_Farchive_" + str(i) + ".dat"), F, i, withArchive=self.withArchive)

                if (self.opt == "max"):
                    fopt.append(max(F))
                    Xopt.append(Xnew[0:len(Xnew) - 1, np.nonzero(F == max(F))[0][0]])
                    if self.withSurr:
                        fopt_byFitnessF.append(max(Ftest))
                else:
                    fopt.append(min(F))
                    Xopt.append(Xnew[0:len(Xnew) - 1, np.nonzero(F == min(F))[0][0]])
                    if self.withSurr:
                        fopt_byFitnessF.append(min(Ftest))

                if (np.mod(i, int(self.niter / 50)) == 0 or i < 1 or self.niter % 1 == 0):
                    stri = "Thread number: {} - evals: {} - max fitness in the generation = {}".format(numThread + 1, i,
                                                                                                       fopt[-1])
                    ###print(stri)
                    with open(self.fileLocSave_exp_num + 'details_threadNum_' + str(numThread + 1) + '.txt',
                              'a') as file:
                        file.write(stri + "\n")

                        # if all(f == self.max_attainable for f in F):
                        #     print(i, "evals: fmax=", fopt, "; done!\n")
                        #     break

            else:
                niter_counter = i
                break

        try:
            self.plot_result(fopt, fopt_byFitnessF, numThread, self.seed)
        except RuntimeWarning:
            print("I am in the warning")

        return fopt, Xopt, self.f_function_counter, niter_counter

    def plot_result(self, fopt, f_opt_real, numThread, seed):
        plt.plot(range(len(fopt)), fopt)
        plt.title("iteration number: {} Thread number {}".format(seed + 1, numThread + 1))
        # plt.show()
        if self.withSurr:
            plt.plot(range(len(f_opt_real)), f_opt_real)
            # plt.show()
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(range(len(fopt)), fopt, label='Fittness using surrogate')
        if self.withSurr:
            ax.plot(range(len(f_opt_real)), f_opt_real, label='Fitness using F function')
        if (self.opt == "max"):
            legend = ax.legend(loc='lower right', shadow=True)
        else:
            legend = ax.legend(loc='upper right', shadow=True)
        plt.title("Fittnes - iteration number: {} Thread number {}".format(seed + 1, numThread + 1))
        plt.savefig(self.fileLocSave_exp_num + "/Graph/fittnes_ThreadNum_" + str(numThread + 1))
        plt.close()

        if self.withSurr:
            plt.plot(range(len(self.surrTest)), self.surrTest)
            plt.title("the absolute value between\n the fittness of the surrogate and the f function")
            # plt.show()
            plt.close()

            plt.plot(range(len(self.surrTest)), self.surrTest)
            plt.title("the absolute value between\n the fittness of the surrogate and the f function")
            plt.savefig(self.fileLocSave_exp_num + "/Graph/diff_ThreadNum_" + str(numThread + 1))
            plt.close()

        plt.close('fig')
        plt.close('all')

    # initialize the experiment
    def init_Exp(self):
        if (self.XInit is None):
            X = self.initCES(self.typeExp)

        else:
            pmut_vector = self.pmut * np.ones(self.popSize, dtype=float)  # individual mutation rates
            X = np.vstack([self.XInit, pmut_vector])

        XARCH = np.array(X[0:self.N, :], dtype=int)  # matrix of the value for each parameter
        St = np.array(X[self.N, :], dtype=float)  # vector of the changing rate for each parameter

        self.write_Exp('exp_Xarchive_0.dat', XARCH, 0, withArchive=self.withArchive)
        self.write_Exp('exp_Sarchive_0.dat', St, 0, withArchive=self.withArchive)

        return XARCH

    def step_Exp(self, numIter):
        if (self.typeExp or numIter == 0):
            ARCH, St, X, F = self.loadDataLocal()
        else:
            # ARCH, St, X, F=self.loadDataFile(numIter)
            ARCH, St, X, F = self.loadDataLocal()

        Xnew, ARCH = self.varCES(X, F, ARCH)  # evaluation steps (mutation, cross over and sexsul selection)
        St = Xnew[self.N, :]
        ARCHtmp = Xnew[0:self.N, :].astype(int)

        # ARCH=ARCH.astype(int)
        # L = np.arange(1, self.popSize + 1).reshape(1, self.popSize)
        # L = np.vstack((L, Xnew[0:self.N, :])).astype(int)
        # self.write_Exp('exp_Xarchive_' + str(numIter) + '.dat', ARCH,numIter)

        # self.write_Exp('exp_Xarchive_' + str(numIter) + '.dat', ARCHtmp, numIter, withArchive=self.withArchive)
        self.write_Exp('exp_Sarchive_' + str(numIter) + '.dat', St, numIter, withArchive=self.withArchive)

        return Xnew

    def shakeThePopulation(self, X, F):
        nBest = math.ceil(self.popSize * 0.05)
        if (self.opt == "max"):
            bestF = heapq.nlargest(nBest, set(F))
        else:
            bestF = heapq.nsmallest(nBest, set(F))
        Xnew = X
        Fnew = F.tolist()
        for i in range(nBest):
            index = Fnew.index(bestF[i])
            Xnew[:, i] = X[:, index]

        for j in range(nBest, self.popSize):
            if (self.repeated_Number):
                Xnew[:, j] = np.random.randint(self.DE[1], size=self.N)
            else:
                ansi = np.arange(self.N)
                np.random.shuffle(ansi)
                Xnew[:, j] = ansi

        return Xnew

    def initCES(self, typeInti):
        X = np.zeros((self.N, self.popSize), dtype=np.int)
        if (typeInti == 1):  # create random data or taking from file
            X = self.initbinary()
        elif self.problem == "QAP":
            X = self.initQap()
        else:
            pass

        pmut_vector = self.pmut * np.ones(self.popSize)  # individual mutation rates
        X = np.vstack([X, pmut_vector])
        return X

    # check if the values combination of the parameters of specific sample is already exist in other sample
    def inArchive(self, x, A):
        if len(A) == 0:
            return False
        else:
            mu = len(A[0])
            res = False
            for i in range(0, mu):
                if (x is A[:, i]):
                    res = True
                    break
        return res

    def loadDataFile(self, numIter):
        ARCH_file = self.fileLocSave_bythread + '/' + 'exp_Xarchive_' + str(numIter - 1) + '.dat'
        ARCH = np.loadtxt(ARCH_file, dtype=int)
        St_file = self.fileLocSave_bythread + '/' + 'exp_Sarchive_' + str(numIter - 1) + '.dat'
        St = np.loadtxt(St_file, dtype=float)
        X = np.vstack((ARCH[0:self.N, len(ARCH[0]) - self.popSize: len(ARCH[0])], St))
        F_file = self.fileLocSave_bythread + '/' + 'exp_Farchive_' + str(numIter - 1) + '.dat'
        F = np.loadtxt(F_file, dtype=float)
        return ARCH, St, X, F

    def loadDataLocal(self):
        ARCH = self.Xarchive
        St = self.Sarchive
        X = np.vstack((ARCH[0:self.N, len(ARCH[0]) - self.popSize: len(ARCH[0])], St))
        F = np.squeeze(self.Farchive)
        return ARCH, St, X, F

    # mutate, cross over and sexual selection
    def varCES(self, X, F, ARCH):
        mu = len(F)

        tau = 1 / (math.sqrt(self.N))
        Xn = np.zeros((self.N + 1, mu))
        Xne = np.zeros((self.N + 1, mu))

        # Elitism - saving the 2 best unique candidates
        _, elitism_index = np.unique(F, return_index=True)
        if(len(elitism_index)<self.num_elitism):
            self.num_elitism=len(elitism_index)

        if (self.opt == "max"):
            elitism_index = elitism_index[(-1 * self.num_elitism):]
        else:
            elitism_index = elitism_index[:self.num_elitism]

        elitism_vec = X[:, elitism_index]

        # Tournament Selection
        for i in range(0, mu):
            parents = np.ceil(
                mu * np.random.rand(self.Tournament_size))  # choose random competitors from the parents generation
            parents = parents.astype(int) - 1
            inew = 0

            if (self.opt == "max"):
                parentsProp = F[parents]
                sumParents = [p / sum(parentsProp) for p in parentsProp]
                sumParents2 = [0 if i < 0 else i for i in sumParents]

                try:
                    inew = np.random.choice(parents, 1, p=sumParents2)
                except ValueError as e:
                    print("Error: {} \nparentes prop: {}".format(e, sumParents2))
            # inew = min(min(np.nonzero(max(F[parents]) == F))) # choose the "best" parents from the q competitors

            else:
                parentsProp = F[parents]
                maxParents = max(parentsProp)
                parentsProp = parentsProp * (-1) + maxParents + 1
                sumParents = [p / sum(parentsProp) for p in parentsProp]
                sumParents2 = [0 if i < 0 else i for i in sumParents]

                try:
                    inew = np.random.choice(parents, 1, p=sumParents2)
                except ValueError as e:
                    print("Error: {} \nparentes prop: {}".format(e, sumParents2))
            # inew = min(min(np.nonzero(min(F[parents]) == F))) # choose the "best" parents from the q competitors

            inew = int(np.squeeze(inew))
            Xn[:, int(i)] = X[:, inew]  # add the chosen parents to the group which will create the next generation

        # Crossover: 1 - point
        for i in range(int(mu / 2)):

            if (random.random() < self.pCrossOver):  # do crossover
                loc = int(math.ceil(self.N * random.random()))  # choose random index to start the crossover process
                if self.repeated_Number:
                    col_tmp = np.hstack(
                        (Xn[0:loc, int(2 * i) + 1], Xn[loc:self.N, int(2 * i)], [Xn[self.N, int(2 * i) + 1]]))
                    Xne[:, int((2 * i))] = col_tmp
                    col_tmp2 = np.hstack(
                        (Xn[0:loc, int(2 * i)], Xn[loc:self.N, int(2 * i) + 1], [Xn[self.N, int(2 * i)]]))
                    Xne[:, int(2 * i) + 1] = col_tmp2
                else:
                    ero_crossOver = ERO(Xn[0:self.N, int(2 * i)], Xn[0:self.N, int(2 * i) + 1])
                    Xne[0:self.N, int(2 * i)], Xne[0:self.N, int(2 * i) + 1] = ero_crossOver.generate_crossover()
                    Xne[self.N, int(2 * i)], Xne[self.N, int(2 * i) + 1] = Xn[self.N, int(2 * i)], Xn[
                        self.N, int(2 * i) + 1]

            else:
                Xne[:, int((2 * i))] = Xn[:, int((2 * i))]
                Xne[:, int(2 * i) + 1] = Xn[:, int(2 * i) + 1]

        # Mutation

        Xnew = np.full((self.N + 1, mu), -1, dtype=float).squeeze()
        for i in range(mu):
            Xnew[self.N, i] = (1 / (1 + ((1 - Xne[self.N, i]) / (Xne[self.N, i])) * (np.exp(-(tau) * (np.random.normal())))))
            if (Xnew[self.N, i] < self.lowest_pmut):
                Xnew[self.N, i] = self.lowest_pmut_defult

            while ((Xnew[0, i]) == -1 | (self.inArchive(Xnew[0:self.N, i], ARCH))):

                if (self.repeated_Number):
                    Xnew[0: self.N, i] = self.CES_Mutate(Xne[0: self.N, i], Xnew[self.N, i], F[i])
                else:
                    Xnew[0: self.N, i] = self.CES_Mutate_combinatoric(Xne[0: self.N, i], Xnew[self.N, i], F[i])

            tmp = (Xnew[0:self.N, i]).reshape(self.N, 1)
            ARCH = np.hstack((ARCH, tmp))


        Xnew[:, 0:self.num_elitism] = elitism_vec
        return Xnew, ARCH

    # mutate the velues
    def CES_Mutate(self, Xold, individual_pmut, f):
        Xnew = np.zeros(self.N)
        for i in range(self.N):
            if (random.random() < individual_pmut):  # we do the mutation in a pmut probability (mutation probability)
                idx = int(math.ceil((self.DE[i]) * random.random()) - 1)
                while (idx == Xold[i]):
                    idx = int(math.ceil((self.DE[i]) * random.random()) - 1)
                Xnew[i] = idx
            else:  # dont mutate i
                Xnew[i] = Xold[i]
        return Xnew

    # mutate the velues
    def CES_Mutate_combinatoric(self, Xold, individual_pmut, f):
        Xnew = np.full((self.N), -1)
        for i in range(self.N):
            if (Xnew[i] < 0):
                if (random.random() < individual_pmut):  # we do the mutation in a pmut probability (mutation probability)
                    idx = int(math.ceil((self.DE[i]) * random.random()) - 1)
                    while (Xnew[idx] > -1):
                        idx = int(math.ceil((self.DE[i]) * random.random()) - 1)
                    Xnew[idx] = Xold[i]
                    Xnew[i] = Xold[idx]

                else:  # dont mutate i
                    Xnew[i] = Xold[i]
        return Xnew

    # evaluate the fitness
    def eval_inExp(self, X, changeCounter):
        ans = np.array(self.fitness(X))
        if changeCounter:
            self.f_function_counter = self.f_function_counter - self.popSize
        return ans

    # crate folder
    def crate_folder(self, folderName):
        folder_Name = os.path.dirname(folderName)
        if not os.path.exists(folder_Name):
            try:
                os.makedirs(os.path.dirname(folderName))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    # write the resulte to a text file
    def write_Exp(self, filename, data, numIter, write=False, withArchive=True):
        if (self.typeExp == 0 or self.typeExp == 1):  # simulation expirement. save the veribles in the main
            if (filename[4] == 'X'):
                if (filename[0] == 'e'):
                    self.Xarchive = data
                    if (withArchive):
                        if (numIter == 0 or len(self.surr_Xarchive) == 0):
                            self.surr_Xarchive = data
                        else:
                            self.surr_Xarchive = np.hstack((self.surr_Xarchive, data))
                    else:
                        if self.sourrogateTyp == "KRIGING":  # in kriging we update the model and not building from scratch every time (so we clean the list to update).
                            self.surr_Xarchive = []
            elif (filename[4] == 'F'):
                self.Farchive = data
                if (withArchive):
                    if (numIter == 0 or len(self.surr_Farchive) == 0):
                        self.surr_Farchive = data
                    else:
                        self.surr_Farchive = np.hstack((self.surr_Farchive, data))
                else:
                    if self.sourrogateTyp == "KRIGING":
                        self.surr_Farchive = []
            else:
                self.Sarchive = data
        if (np.mod(numIter, float(self.niter / 40)) == 0 or write or numIter % 1 == 0):
            fid = open(self.fileLocSave_bythread + '/' + filename, 'w')
            if (data.ndim == 1):  # if the data is vector - wirte in 1 line
                tmp = str(data)
                tmp = tmp.replace("\n", "")
                tmp = tmp.replace("\n", "")
                fid.write(tmp[1:len(tmp) - 1])
            else:  # if the data is metrix, write each row in different row
                for i in range(0, len(data)):
                    tmp = str(data[i])
                    tmp = tmp.replace("\n", "")
                    fid.write(tmp[1:len(tmp) - 1] + "\n")
            fid.close()
        lock2.acquire()
        if not os.path.exists(self.fileLocSave_exp_num + "/Graph"):
            os.makedirs(self.fileLocSave_exp_num + "/Graph")
        lock2.release()

    def initQap(self):
        ans = np.zeros((self.N, self.popSize))
        for j in range(self.popSize):
            ansi = np.arange(self.N)
            np.random.shuffle(ansi)
            ans[:, j] = ansi
        return ans

    def initbinary(self):
        ans = np.zeros((self.N, self.popSize))
        for i in range(0, self.popSize):  # initialize for each parameter in each sample its primary (random) value
            for j in range(self.N):
                ans[j][i] = int(math.ceil(self.DE[j] * random.random()) - 1)
        return ans


def ONEMAX(X):
    mu = len(X[0])
    fitness = []
    for i in range(mu):
        fitness.append(sum(X[:, i]))
    return fitness


def tmp(X):
    X = X + 1
    local_state = np.random.RandomState(1)
    return local_state.rand(1, 24) * 5


def SwedishPumpFit(X):
    """ The correlation function, assumes a numpy vector {-1,+1} as input """
    X[X == 0] = -1
    ans = []
    for i in range(len(X[0])):
        n = len(X[:, i])
        E = []
        for k in range(1, n):
            X1 = X[0:n - k, i]
            X2 = X[k:, i]
            X3 = X1.dot(X2)
            E.append(X3 ** 2)

            # E.append((X[0:n - k, i].dot(X[k:, i])) ** 2)
        ans.append((float(n ** 2) / float(2 * sum(E))))
    return ans


def NKLandscapeFit(X):
    X = X.astype(int)
    ans = []
    with open("/home/naamah/Documents/CatES/result_All/NKL/init_NKL.p", "rb") as fp:
        model = pickle.load(fp)
    # model = NKlandscape(len(X), 5)
    for x in range(len(X[0])):
        tmp = np.array_str(X[:, x]).replace('\n', '')
        tmp = tmp[1:len(tmp) - 1].replace('\n', '')
        tmp = "".join(tmp.split())

        ans.append(model.compFit(tmp))
    return ans


def QAPFit(X):
    # github: https://github.com/danielgribel/qap/tree/master/data
    X = X.astype(int)
    distance, flow = read_file(len(X))

    sum = 0
    size = len(X)
    for i in range(0, size):
        for j in range(0, size):
            if i != j:
                x = flow[X[i], X[j]] * distance[i, j]
                sum = sum + x
    ans = sum.tolist()
    return ans


def read_file(n):
    ins = open("/home/naamah/Documents/CatES/data/nug12.dat", "r")
    i = 0
    br = 0
    rowdistance = 0
    rowdflow = 0
    tmpdistance = np.empty((n, n))
    tmpflow = np.empty((n, n))

    for line in ins:
        if i == 0:
            x = line.split("\n")[0]
            tmpdistance = np.empty((int(x), int(x)))
            tmpflow = np.empty((int(x), int(x)))
        elif line == "\n":
            br += 1
        else:
            if br == 1:
                tmpList = line.split(" ")
                tmpList = tmpList[0:len(tmpList)]
                numbers = [int(y) for y in tmpList]
                tmpdistance[rowdistance] = np.asarray(numbers)
                rowdistance = rowdistance + 1

            if br == 2:
                indexFlow = 0
                for k in re.split(' +', line):
                    if k != '':
                        tmpflow[rowdflow][indexFlow] = int(k.split("\n")[0])
                        indexFlow = indexFlow + 1
                rowdflow = rowdflow + 1
        i += 1

    return tmpdistance, tmpflow


class InitProblem():
    def NKLandscape(self):
        typeExp = 1  # lab (0) or simulation (1)
        numPar = 20  # number of dominations / parameters
        popSize = 30  # number of sample in each generation
        fitness = NKLandscapeFit  # the fitness function
        DEVector = [0,
                    2]  # number of levels for each dimnation. the index 0 will represent all the dimnation have the same lavel (0=same lavel, 1= different)
        max_attainable = np.inf
        opt = 'max'
        repeated_Number = True  # is the number in each vector in the population can repeted?

        # initialize the NKlandscape model - only if need to change the model!!!!!!!!!!!!
        # model = NKlandscape(numPar, 2)
        # with open("/home/naamah/Documents/CatES/result_All/NKL/init_NKL.pickle", "wb") as fp:  # Pickling
        #    pickle.dump(model, fp, protocol=2)


        return typeExp, numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number

    def QAP(self):
        typeExp = 0  # lab (0) or simulation (1)
        numPar = 12  # number of dominations / parameters
        popSize = 30  # number of sample in each generation
        fitness = QAPFit  # the fitness function
        DEVector = [
            1]  # number of levels for each dimnation. the index 0 will represent all the dimnation have the same lav el (0=same lavel, 1= different)
        for i in range(popSize):
            DEVector.append(numPar)
        max_attainable = np.inf
        opt = 'min'
        repeated_Number = False
        return typeExp, numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number

    def protein(self):
        typeExp = 1  # lab (0) or simulation (1)
        numPar = 10  # number of dominations / parameters
        popSize = 24  # number of sample in each generation
        fitness = tmp  # the fitness function
        DEVector = [1, 4, 5, 3, 6, 5, 4, 7, 2, 11, 3]  # number of levels for each dimnation the index 0 will represent
        # if it the same lavel (0=same lavel, 1= different)
        max_attainable = np.inf
        opt = 'max'
        repeated_Number = True
        return typeExp, numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number

    def maxOne(self):
        typeExp = 1  # lab (0) or simulation (1)
        numPar = 100
        popSize = 50
        fitness = ONEMAX
        DEVector = [0, 2]
        max_attainable = 100
        opt = 'max'
        repeated_Number = True
        return typeExp, numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number

    def swedishPump(self):
        typeExp = 1  # lab (0) or simulation (1)
        numPar = 50
        popSize = 30
        fitness = SwedishPumpFit
        DEVector = [0, 2]
        max_attainable = np.inf
        opt = 'max'
        repeated_Number = True
        return typeExp, numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number


def main(pmut=0.4, lowest_pmut=0.005, lowest_pmut_defult=0.01, pCrossOver=0.5, Tournament_size=10, num_elitism=2,
         problems_list="NaN", sourrogates_list="NaN", FileNumToSave=100, numThreads=2, XInit=None,
         criterion="mse", max_depth=51646, max_features="auto",max_leaf_nodes=26, min_impurity_decrease=0.0349,
         min_samples_leaf=16, min_samples_split=5,min_weight_fraction_leaf=0.4662, n_estimators=17,warm_start=False,
         n_neighbors=9, algorithm="auto", p=1, weights="distance",leaf_size=5,
         kernel='rbf', C=1000, epsilon=0.0012569742906175072, gamma=0.04451123874003816, shrinking=True,degree=2,coef0=0.1):

    problem_list = [problems_list]
    sourrogate_list = [sourrogates_list]

    if (problems_list == "NaN"):
        problem_list = ["Pump"]  # "Pump","NKL","QAP"

    if sourrogates_list == "NaN":
        sourrogate_list = ["SVM"]  # RBFN KRIGING "RF","SVM","KNN"


    RF_parm={"criterion":criterion, "max_depth":max_depth, "max_features":max_features,"max_leaf_nodes":max_leaf_nodes,"min_impurity_decrease":min_impurity_decrease, "min_samples_leaf":min_samples_leaf,
             "min_samples_split":min_samples_split,"min_weight_fraction_leaf":min_weight_fraction_leaf, "n_estimators":n_estimators,"warm_start":warm_start}
    knn_parm={"n_neighbors":n_neighbors, "algorithm":algorithm, "p":p, "weights":weights,"leaf_size":leaf_size}
    svm_parm={"kernel":kernel, "C":C, "epsilon":epsilon, "gamma":gamma,"shrinking":shrinking,"degree":degree,"coef0":coef0}

    for problem in problem_list:
        for sourrogate in sourrogate_list:
            probInit = InitProblem()
            numThreads = int(numThreads)
            niter = 32
            FileNumToSave = int(FileNumToSave)
            withSurr = True
            withThreads = True
            if withSurr:
                numToCompare = 2
            else:
                numToCompare=1

            fileName = sourrogate + "_" + str(FileNumToSave)
            if problem == "Pump":
                typeExp, numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number = probInit.swedishPump()
            elif problem == "QAP":
                typeExp, numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number = probInit.QAP()
            else:  # NKL
                typeExp, numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number = probInit.NKLandscape()

            call_F_function_counter = popSize * 70  # 2100
            ans = [[] for i in range(numToCompare)]
            for seed in range(numToCompare):
                start_time = datetime.now()
                psutil.cpu_percent(interval=None)
                psutil.cpu_times_percent(interval=None)
                avr = 0

                if seed == 1:
                    withSurr = False
                cat = CatES(fitness, DEVector, typeExp, numPar, popSize, seed, max_attainable, niter, opt,
                            repeated_Number, sourrogate,
                            FileNumToSave, call_F_function_counter, withSurr, problem, pmut, lowest_pmut,
                            lowest_pmut_defult, pCrossOver, Tournament_size, num_elitism, XInit,RF_parm,svm_parm,knn_parm)

                # cat.set_exp()
                ###print("iteration number {}".format(seed + 1))
                fmax = []
                fmaxSum = 0
                numIter = niter
                num_call_F = call_F_function_counter
                maxFitList = []
                maxVectorList = []

                if withThreads:
                    fmax.append(
                        Parallel(n_jobs=numThreads, verbose=0)(delayed(cat.test_exp)(i) for i in range(numThreads)))
                else:
                    fmax.append(cat.test_exp(numThreads))
                    if (opt == 'min'):
                        return min(fmax[0][0])
                    else:
                        return max(fmax[0][0])

                for i in range(numThreads):
                    maxFitList.append(fmax[0][i][0][-1])
                    maxVectorList.append(fmax[0][i][1][-1])
                    fmaxSum = fmaxSum + fmax[0][i][0][-1]
                num_call_F = num_call_F - fmax[0][0][2]
                numIter = fmax[0][0][3]
                avr = avr + fmaxSum
                ans[seed].append(maxFitList)

                if (opt == 'min'):
                    index = maxFitList.index(min(maxFitList))
                    bestFitt = min(maxFitList)
                else:
                    index = maxFitList.index(max(maxFitList))
                    bestFitt = max(maxFitList)

                bestVector = str(maxVectorList[index])[1:-1].replace("\n", "")
                bestVector = bestVector.replace("          ", ",")
                bestVector = bestVector.replace("  ", ",")
                ###print("\naverage of max fitness in {} different runs is: {}".format(numThreads,(avr / numThreads)))
                ###print("\nBest fitness is {} with vector: {}".format(bestFitt, bestVector))

                # general details
                with open("/home/naamah/Documents/CatES/result_All/" + problem + "/" + fileName + "/General_Info.txt",
                          'a') as file:
                    if withSurr:
                        file.write("\b With Surrogate model \b" + "\n")
                    else:
                        file.write('\b Without Surrogate model \b' + '\n')

                    end_time = datetime.now()
                    stri = 'Duration: {}'.format(end_time - start_time)
                    cpu_percent = "cpu percent: {}".format(str(psutil.cpu_percent(interval=None)))
                    cpu_time = "cpu time: {}".format(str(psutil.cpu_times_percent(interval=None)))

                    file.write("Num of Threads: {}".format(numThreads) + "\n")
                    file.write("Num of Iterations (for each Thread): {}".format(numIter) + "\n")
                    file.write("Num of calls to the F functions (for each Thread): {}".format(num_call_F) + "\n")
                    file.write(stri + "\n")
                    file.write(cpu_percent + "\n")
                    # file.write(cpu_time + "\n")
                    file.write("Avrage Fittness: {}".format(avr / numThreads) + "\n")
                    file.write("Best Fittness: {}".format(bestFitt) + "\n")
                    file.write("Best Vector: {}".format(bestVector) + "\n\n\n")

                    ###print(stri)
                    ###print(cpu_percent)
                file.close()

            if numToCompare>1:
                print("ans: {}".format(ans))
                a = np.asarray(ans[0])
                b = np.asarray(ans[1])

                twosample_results = stats.ttest_ind(a[0], b[0])

                # significate
                print("\n\n***  Stats  ***")
                print("t: {} \np: {}".format(twosample_results[0], twosample_results[1]))
                with open("/home/naamah/Documents/CatES/result_All/" + problem + "/" + fileName + "/General_Info.txt",
                          'a') as file:
                    file.write("T test: {}".format(twosample_results[0]) + "\n")
                    file.write("P value: {}".format(twosample_results[1]) + "\n")
                file.close()

                # boxplot
                try:
                    data = [a[0], b[0]]
                    fig, axs = plt.subplots(1, 2, sharey=True)
                    plt.suptitle("{} - {} \n P Value: {}".format(problem, sourrogate, twosample_results[1]))

                    axs[0].boxplot(a[0])
                    axs[0].set_title('with Sorrugate')
                    plt.ylim(min(min(a[0]), min(b[0])) * 0.98, max(max(a[0]), max(b[0])) * 1.02)
                    axs[1].boxplot(b[0])
                    axs[1].set_title("without Sorrugate")

                    plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)

                    plt.savefig("/home/naamah/Documents/CatES/result_All/" + problem + "/" + fileName + "/BoxPlot")
                    # plt.show()
                    plt.close('all')
                    plt.close(fig)
                except RuntimeWarning:
                    print("i am in the warning")
    return bestFitt


if __name__ == '__main__':
    lock2 = Lock()
    FileNumToSave = 123
    niter = 100
    numThreads = 30
    #print("main ans: {}".format(main(FileNumToSave=FileNumToSave,numThreads=numThreads)))
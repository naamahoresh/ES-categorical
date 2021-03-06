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
from pyDOE import *
import dexpy.factorial as factorial
from rpy2.robjects.packages import importr
FrF2 = importr('FrF2')
lhs_R = importr('lhs')
import dexpy.optimal
import sys
import shutil
import glob
import ghalton
sys.path.insert(0, '/home/naamah/Documents/CatES/')
import os.path

from NKLandscape_class import NKlandscape
from EdgeCrossOver import ERO
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)
import surrogate as surrogate
from datetime import datetime
lock2 = Lock()
from Cat_problems import InitProblem
import statistics

class CatES:
    def __init__(self, fitnessfun, DEVector, typeExp, numPar, popSize, seed=0, max_attainable=np.inf, niter=100,
                 opt='max', repeated_Number=False, sorrugate="KRIGING", f_function_counter=2100,
                 withSurr=True, problem="QAP", intiNumIterWituoutSurr=20, isKmeans=True, exp_condition="Surrogate",
                 DOE_popsize=0,fileLoc_general=""):
        self.numThread=0
        # if problem in ["Ising2DSquare", "IsingTriangle"]:
        #     self.N = numPar*numPar  # number of parameters
        # else:
        self.N=numPar
        self.popSize = popSize  # number of sample for each generation
        self.niter = niter  # number iterations
        self.typeExp = typeExp;  # lab expirance or simulation. 0 = lab. 1 = simulation
        self.fitness = fitnessfun
        # self.local_state = np.random.RandomState(seed)
        self.fileLocSave_general = fileLoc_general
        if (exp_condition=="LHS" or exp_condition=="FFD" or exp_condition=="DOP" or exp_condition=="HLT"  ):
            self.fileLocSave_exp_num = self.fileLocSave_general + "/exp_num_" + str(seed + 1) +"_"+ exp_condition +"_"+str(int(DOE_popsize*100/f_function_counter))+ "/"
        else:
            self.fileLocSave_exp_num = self.fileLocSave_general + "/exp_num_" + str(seed + 1) + "_" + exp_condition+ "/"

        self.fileLocSave_IOHProfiler = self.fileLocSave_general+"/IOHProfiler/"

        self.fileLocSave_bythread = self.fileLocSave_exp_num
        self.exp_number_condition = "/exp_num_" + str(seed + 1) + "_" + exp_condition
        self.max_attainable = max_attainable
        self.opt = opt
        self.Farchive = np.empty(0)
        self.Xarchive = np.empty(0)
        self.Sarchive = np.empty(0)
        self.repeated_Number = repeated_Number
        self.withArchive = True
        self.sorrugateTyp = sorrugate
        self.f_function_counter = f_function_counter
        self.withSurr = withSurr
        self.problem = problem
        self.creat_folder(self.fileLocSave_exp_num)
        self.creat_folder(self.fileLocSave_IOHProfiler)
        self.seed = seed
        self.intiNumIterWituoutSurr=intiNumIterWituoutSurr
        self.isKmeans=isKmeans
        self.surr_Farchive = []
        self.surr_Xarchive = []
        self.surrTest = []
        self.exp_condition=exp_condition
        if (DOE_popsize==0):
            self.DOE_popsize = popSize
        else:
            self.DOE_popsize = DOE_popsize

        self.pmut = 0.1183
        self.Tournament_size = 11
        self.num_elitism =6
        self.lowest_pmut =0.0443
        self.lowest_pmut_defult =0.0929
        self.pCrossOver=0.2221
        if (problem=="LABS" and sorrugate=="KNN"):
            self.pmut = 0.1147
            self.Tournament_size = 5
            self.num_elitism = 10
            self.lowest_pmut = 0.0193
            self.lowest_pmut_defult = 0.087
            self.pCrossOver = 0.7976


        # inti DE - number of levels for each parameter. 2 options:
        # 1. all the parameters have same level
        # 2. each parameter have different level
        if (DEVector[0] == 0):
            self.DE = np.full((1, self.N), DEVector[1]).squeeze()
        else:
            self.DE = np.array(DEVector[1:self.N + 1])

        with open(self.fileLocSave_general + "/ES_configuration.txt", 'a') as file:
            file.write("Thread num: {}\n pmut = {}\n lowest_pmut = {}\n lowest_pmut_defult = {}\n pCrossOver = {}\n Tournament_size = {}\n num_elitism = {}\n\n\n".
                       format(str(seed + 1), self.pmut, self.lowest_pmut, self.lowest_pmut_defult, self.pCrossOver, self.Tournament_size, self.num_elitism))
        file.close()


    def set_exp(self):
        iter = 0
        if (not iter):
            Xnew = self.init_Exp()
        else:
            Xnew = self.step_Exp(iter)

    def init_folder_by_thread(self,numThread):
        self.numThread =numThread
        self.fileLocSave_bythread = self.fileLocSave_bythread + "threadNum_" + str(numThread) + "/"
        self.creat_folder(self.fileLocSave_bythread)
        self.creat_folder(self.fileLocSave_IOHProfiler+self.exp_number_condition+'/')
        # self.creat_folder(self.fileLocSave_IOHProfiler+"exp_" + self.exp_number_condition[self.exp_number_condition.find("num_")+4] + "_" + self.exp_condition)
        # self.creat_folder(self.fileLocSave_IOHProfiler+"exp_" + self.exp_number_condition[self.exp_number_condition.find("num_")+4] + "_" + self.exp_condition+ "/data_f1")

        # create file to the IOHProfiler
        with open(self.fileLocSave_IOHProfiler+self.exp_number_condition+'/profiler_format_T' + str(numThread) + '.dat', 'a') as file:
            tmp_str = '"function evaluation" "current f(x)" "best-so-far f(x)" "current af(x)+b"  "best af(x)+b" '
            file.write(tmp_str+" \n")

    def fitness_by_surrogate(self,Xnew, init_model):
        if self.sorrugateTyp == "RBFN":
            F = surrogate.surrogateRBFN(self.surr_Xarchive[:, 0:len(self.surr_Farchive)],
                                        self.surr_Farchive,
                                        Xnew[0:self.N, :], self.fileLocSave_bythread,
                                        self.fileLocSave_general, self.withArchive, False,
                                        self.problem, isKmeans=self.isKmeans)  # call to the soraggate model
        elif self.sorrugateTyp == "KRIGING":
            if init_model:
                F = surrogate.surrogateKriging(self.surr_Xarchive[:, 0:len(self.surr_Farchive) + 1],
                                               self.surr_Farchive, Xnew[0:self.N, :], self.fileLocSave_bythread,
                                               self.fileLocSave_general, self.withArchive, True, self.problem)
            else:
                F = surrogate.surrogateKriging(self.surr_Xarchive, self.surr_Farchive,
                                           Xnew[0:self.N, :], self.fileLocSave_bythread,
                                           self.fileLocSave_general, self.withArchive, False,
                                           self.problem)

        elif self.sorrugateTyp == "RF":
            F = surrogate.surrogateRF(self.surr_Xarchive, self.surr_Farchive,
                                      Xnew[0:self.N, :], self.fileLocSave_bythread,
                                      self.fileLocSave_general, self.withArchive, False,
                                      self.problem)  # call to the soraggate model

        elif self.sorrugateTyp == "KNN":
            F = surrogate.surrogateKNN(self.surr_Xarchive, self.surr_Farchive,
                                       Xnew[0:self.N, :], self.fileLocSave_bythread,
                                       self.fileLocSave_general, self.withArchive, False,
                                       self.problem)  # call to the soraggate model

        else:
            F = surrogate.surrogateSVM(self.surr_Xarchive, self.surr_Farchive,
                                       Xnew[0:self.N, :], self.fileLocSave_bythread,
                                       self.fileLocSave_general, self.withArchive, False,
                                       self.problem)  # call to the soraggate model
        return F

    def test_exp(self, numThread):
        self.init_folder_by_thread(numThread)
        fopt,fopt_by_real_fintess,Xopt = [],[],[]

        X = self.init_Exp()
        F = self.eval_inExp(X, True)
        self.withArchive = True
        self.write_Exp("Farchive",'exp_Farchive_0.dat', F, 0, withArchive=self.withArchive)
        iter_counter = self.niter

        if (self.f_function_counter<=0):
            stri = "Thread number: {} - eval: {} - max fitness in the generation = {}".format(numThread, 0, max(F))
            with open(self.fileLocSave_exp_num + 'details_threadNum_' + str(numThread) + '.txt', 'a') as file:
                file.write(stri + "\n")
            self.write_profiler_format(numThread, max(F), 0)
            return F, X, self.f_function_counter, self.niter

        for i in range(1, self.niter + 1):  # num iterations
            if self.f_function_counter > 0:  # num calls to the F function
                Xnew = self.step_Exp(i)

                if self.withSurr:  # exp with surrogate
                    if (i < self.intiNumIterWituoutSurr or i % 9 == 0 or i % 10 == 0):
                        F = self.eval_inExp(Xnew[0:len(Xnew) - 1, :], True)
                        self.withArchive = True
                        F_real_fitness = F

                    else:
                        F = self.fitness_by_surrogate(Xnew, i==self.intiNumIterWituoutSurr+1)
                        F_real_fitness = self.eval_inExp(Xnew[0:len(Xnew) - 1, :], False)
                        self.surrTest.append(np.linalg.norm(F_real_fitness - F))
                        self.withArchive = False

                else:  # exp without surrogate
                    F = self.eval_inExp(Xnew[0:len(Xnew) - 1, :], True)

                self.write_Exp("Xarchive",'exp_Xarchive_' + str(i) + '.dat', Xnew[0:self.N, :], self.niter,withArchive=self.withArchive)
                self.write_Exp("Farchive",str("exp_Farchive_" + str(i) + ".dat"), F, i, withArchive=self.withArchive)

                if (self.opt == "max"):
                    fopt.append(max(F))
                    Xopt.append(Xnew[0:len(Xnew) - 1, np.nonzero(F == max(F))[0][0]])
                    if self.withSurr:
                        fopt_by_real_fintess.append(max(F_real_fitness))
                else:
                    fopt.append(min(F))
                    Xopt.append(Xnew[0:len(Xnew) - 1, np.nonzero(F == min(F))[0][0]])
                    if self.withSurr:
                        fopt_by_real_fintess.append(min(F_real_fitness))

                stri = "Thread number: {} - eval: {} - max fitness in the generation = {}".format(numThread, i,
                                                                                                  fopt[-1])
                # print(stri)
                with open(self.fileLocSave_exp_num + 'details_threadNum_' + str(numThread) + '.txt', 'a') as file:
                    file.write(stri + "\n")

                    if all(f == self.max_attainable for f in F):
                        print(i, "evals: fmax=", fopt, "; done!\n")
                        break

                self.write_profiler_format(numThread , fopt[-1],i)

            else:
                iter_counter = i-1
                break

        try:
            self.plot_result(fopt, fopt_by_real_fintess, numThread, self.seed)
        except RuntimeWarning:
            print("Can't plot result")

        return fopt, Xopt, self.f_function_counter, iter_counter


    def write_profiler_format(self, numThread, fittnes, iter):
        with open(self.fileLocSave_general + '/IOHProfiler'+self.exp_number_condition+'/profiler_format_T' + str(numThread) + '.dat', 'a') as file:
            file.write(str(iter)+" "+str(fittnes)+" "+str(fittnes)+" "+str(fittnes)+" "+str(fittnes)+" \n")

    def plot_result(self, fopt, f_opt_real, numThread, seed):
        plt.plot(range(len(fopt)), fopt)
        plt.title("iteration number: {} Thread number {}".format(seed + 1, numThread))
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
        plt.title("Fittnes - iteration number: {} Thread number {}".format(seed + 1, numThread))
        plt.savefig(self.fileLocSave_exp_num + "/Graph/fittnes_ThreadNum_" + str(numThread))
        plt.close()

        if self.withSurr:
            plt.plot(range(len(self.surrTest)), self.surrTest)
            plt.title("the absolute value between\n the fittness of the surrogate and the f function")
            # plt.show()
            plt.close()

            plt.plot(range(len(self.surrTest)), self.surrTest)
            plt.title("the absolute value between\n the fittness of the surrogate and the f function")
            plt.savefig(self.fileLocSave_exp_num + "/Graph/diff_ThreadNum_" + str(numThread ))
            plt.close()

        plt.close('fig')
        plt.close('all')

    # initialize the experiment
    def init_Exp(self):

        if (self.exp_condition=="LHS"):
            X =self.initLHS()
            # self.f_function_counter-=self.DOE_popsize

        elif (self.exp_condition=="FFD"):
            X =self.initFF()
            # self.f_function_counter-=self.DOE_popsize

        elif  (self.exp_condition=="DOP"):
            X =self.initDopt()
            # self.f_function_counter-=self.DOE_popsize

        elif  (self.exp_condition=="HLT"):
            X =self.initHalton()
            # self.f_function_counter-=self.DOE_popsize

        else:
            X = self.initCES()
        #
        # XARCH = np.array(X[0:self.N, :], dtype=int)  # matrix of the value for each parameter
        # St = np.array(X[self.N, :], dtype=float)  # vector of the changing rate for each parameter

        XARCH = np.array(X[0:len(X)-1, :], dtype=int)  # matrix of the value for each parameter
        St = np.array(X[len(X)-1, :], dtype=float)  # vector of the changing rate for each parameter

        self.write_Exp("Xarchive",'exp_Xarchive_0.dat', XARCH, 0, withArchive=self.withArchive)
        self.write_Exp("Sarchive",'exp_Sarchive_0.dat', St, 0, withArchive=self.withArchive)

        return XARCH

    def step_Exp(self, numIter):
        if (self.typeExp or numIter == 0):
            ARCH, St, X, F = self.loadDataLocal(numIter)
        else:
            # ARCH, St, X, F=self.loadDataFile(numIter)
            ARCH, St, X, F = self.loadDataLocal(numIter)

        Xnew, ARCH = self.varCES(X, F, ARCH)  # evaluation steps (mutation, cross over and sexsul selection)

        St = Xnew[self.N, :]
        ARCHtmp = Xnew[0:self.N, :].astype(int)

        # ARCH=ARCH.astype(int)
        # L = np.arange(1, self.popSize + 1).reshape(1, self.popSize)
        # L = np.vstack((L, Xnew[0:self.N, :])).astype(int)
        # self.write_Exp('exp_Xarchive_' + str(numIter) + '.dat', ARCH,numIter)

        # self.write_Exp("Xarchive",'exp_Xarchive_' + str(numIter) + '.dat', ARCHtmp, numIter, withArchive=self.withArchive)
        self.write_Exp("Sarchive",'exp_Sarchive_' + str(numIter) + '.dat', St, numIter, withArchive=self.withArchive)

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
                random.shuffle(ansi)
                Xnew[:, j] = np.asarray(ansi)

        return Xnew

    def initCES(self):
        if self.problem == "QAP":
            X = self.initQap()
        else:
            X=self.initbinary()


        pmut_vector = self.pmut * np.ones(self.popSize)  # individual mutation rates
        X = np.vstack([X, pmut_vector])
        return X

    def initLHS(self):
        if self.problem == "QAP":
            print("Error - QAP LHS Not implemented yet")
        else:
            np.random.seed()
            X = lhs(n=self.N,samples=self.DOE_popsize,criterion='c')
            X = X.transpose()
            # X =np.asarray(lhs_R.randomLHS(self.N,self.DOE_popsize))
            X = np.round(X)

        pmut_vector = self.pmut * np.ones(self.DOE_popsize)  # individual mutation rates
        X = np.vstack([X, pmut_vector])
        return X

    def initHalton(self):
        if self.problem == "QAP":
            print("Error - QAP Halton Not implemented yet")
        else:
            random.seed()
            sequencer = ghalton.GeneralizedHalton(self.N,random.randint(0, 1000000))
            X =np.asarray(sequencer.get(self.DOE_popsize))
            X=X.transpose()
            X = np.round(X)
        pmut_vector = self.pmut * np.ones(self.DOE_popsize)  # individual mutation rates
        X = np.vstack([X, pmut_vector])
        return X

    def initDopt(self):
        sys.setrecursionlimit(1500)
        if(self.DOE_popsize<self.N):
            self.DOE_popsize=self.N+1
        model_par = []
        for i in range(1,self.N+1): #creat first-order mordl (X1+X2+..+XNn)
            model_par.append(''.join(["X", str(i)]))
        model = '(' + '+'.join(model_par) + ')'

        X = dexpy.optimal.build_optimal(factor_count=self.N, run_count=self.DOE_popsize, model=model)
        X = np.asarray(X,dtype=int)
        X[X == -1] = 0
        X=X.transpose()

        pmut_vector = self.pmut * np.ones(self.DOE_popsize)  # individual mutation rates
        X = np.vstack([X, pmut_vector])
        return X


    def initFF(self):
        if self.problem == "QAP":
            print("Error - QAP FFD Not implemented yet")
        else:
            self.DOE_popsize = round(math.log(self.DOE_popsize)/math.log(2))
            while(2**self.DOE_popsize <= self.N):
                self.DOE_popsize+=1
            x_tmp=FrF2.FrF2(nruns=2**self.DOE_popsize,nfactors=self.N, randomize=False)
            X = np.asarray(x_tmp)#MAX: 2**12
            X[X == 2] = 0
            # X[:, -1]=np.random.randint(2, size=len(X))
            # X[:, 0]=np.random.randint(2, size=len(X))

            # X = np.asarray(factorial.build_factorial(self.N, 2**self.DOE_popsize)) # variable limit
            # X[X == -1] = 0
            # X= X.transpose()

        pmut_vector = self.pmut * np.ones(len(X[0]))  # individual mutation rates
        X = np.vstack([X, pmut_vector])
        return X

    def initQap(self):
        ans = np.zeros((self.N, self.popSize))
        for j in range(self.popSize):
            ansi = np.arange(self.N)
            random.shuffle(ansi)
            ans[:, j] = np.asarray(ansi)
        return ans

    def initbinary(self):
        ans = np.zeros((self.N, self.popSize))
        for i in range(0, self.popSize):  # initialize for each parameter in each sample its primary (random) value
            for j in range(self.N):
                ans[j][i] = int(math.ceil(self.DE[j] * random.random()) - 1)
        return ans

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

    def loadDataLocal(self,numIter):
        ARCH = self.Xarchive
        St = self.Sarchive

        if ((self.exp_condition=="LHS" or self.exp_condition=="FFD" or self.exp_condition=="DOP" or self.exp_condition=="HLT") and numIter==1 ):
            # X = np.vstack((ARCH[0:self.N, len(ARCH[0]) - self.DOE_popsize:len(ARCH[0])], St))
            X = np.vstack((ARCH[0:self.N,:], St))

        else:
            X = np.vstack((ARCH[0:self.N, len(ARCH[0]) - self.popSize: len(ARCH[0])], St))
        F = np.squeeze(self.Farchive)
        return ARCH, St, X, F

    # mutate, cross over and sexual selection
    def varCES(self, X, F, ARCH):
        tau = 1 / (math.sqrt(self.N))

        elitism_vec = self.elitism(X,F)
        Xn=self.tourment(X, F, len(F))
        Xne=self.cross_over(Xn)
        Xnew,tmp=self.mutation(Xne,F, ARCH,tau)

        ARCH = np.hstack((ARCH, tmp))
        Xnew[:, 0:self.num_elitism] = elitism_vec

        return Xnew, ARCH


    def tourment(self,X,F,mu):
        Xn = np.zeros((self.N + 1, self.popSize))
        for i in range(0, self.popSize):
            parents = np.ceil(mu * np.random.rand(self.Tournament_size))  # choose random competitors from the parents generation
            parents = parents.astype(int) - 1
            inew = 0

            if (self.opt == "max"):
                parentsProp = F[parents]
                minParent = min(parentsProp)
                if minParent<0:
                    parentsProp+=(minParent*-1)
                sumParents = [0 if i < 0 else i for i in parentsProp]
                sumParents2 = [p / sum(sumParents) for p in sumParents]

                try:
                    inew = np.random.choice(parents, 1, p=sumParents2)
                except ValueError as e:
                    print("Error: {} \nparentes prop: {}".format(e, sumParents2))
            # inew = min(min(np.nonzero(max(F[parents]) == F))) # choose the "best" parents from the q competitors

            else:
                parentsProp = F[parents]
                maxParents = max(parentsProp)
                parentsProp = parentsProp * (-1) + maxParents + 1
                sumParents = [0 if i < 0 else i for i in parentsProp]
                sumParents2 = [p / sum(sumParents) for p in sumParents]

                try:
                    inew = np.random.choice(parents, 1, p=sumParents2)
                except ValueError as e:
                    print("Error: {} \nparentes prop: {}".format(e, sumParents2))
            # inew = min(min(np.nonzero(min(F[parents]) == F))) # choose the "best" parents from the q competitors

            inew = int(np.squeeze(inew))
            Xn[:, int(i)] = X[:, inew]  # add the chosen parents to the group which will create the next
        return Xn

    def elitism(self,X,F):
        _, elitism_index = np.unique(F, return_index=True)
        if (len(elitism_index) <self.num_elitism):
            self.num_elitism = len(elitism_index)
        if (self.opt == "max"):
            elitism_index = elitism_index[(-1 * self.num_elitism):]
        else:
            elitism_index = elitism_index[:self.num_elitism]
        elitism_vec = X[:, elitism_index]
        return elitism_vec


    def cross_over(self,Xn):
        Xne = np.zeros((self.N + 1, self.popSize))
        for i in range(int(self.popSize / 2)):

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
                    Xne[self.N, int(2 * i)], Xne[self.N, int(2 * i) + 1] = Xn[self.N, int(2 * i)], Xn[self.N, int(2 * i) + 1]

            else:
                Xne[:, int((2 * i))] = Xn[:, int((2 * i))]
                Xne[:, int(2 * i) + 1] = Xn[:, int(2 * i) + 1]

        return Xne


    def mutation(self,Xne,F, ARCH,tau):
        Xnew = np.full((self.N + 1, self.popSize), -1, dtype=float).squeeze()
        for i in range(self.popSize):
            Xnew[self.N, i] = (1 / (1 + ((1 - Xne[self.N, i]) / (Xne[self.N, i])) * (np.exp(-(tau) * (np.random.normal())))))
            if (Xnew[self.N, i] < self.lowest_pmut):
                Xnew[self.N, i] = self.lowest_pmut_defult
            while ((Xnew[0, i]) == -1 | (self.inArchive(Xnew[0:self.N, i], ARCH))):
                if (self.repeated_Number):
                    Xnew[0: self.N, i] = self.CES_Mutate(Xne[0: self.N, i], Xnew[self.N, i])
                else:
                    Xnew[0: self.N, i] = self.CES_Mutate_combinatoric(Xne[0: self.N, i], Xnew[self.N, i])

            tmp = (Xnew[0:self.N, i]).reshape(self.N, 1)
        return Xnew, tmp


    def CES_Mutate(self, Xold, individual_pmut):
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

    def CES_Mutate_combinatoric(self, Xold, individual_pmut):
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

    def eval_inExp(self, X, change_fitness_funcation_counter):
        ans = np.array(self.fitness(X))
        if change_fitness_funcation_counter:
            self.f_function_counter -= len(ans)
        return ans

    def creat_folder(self, folderName):
        path_name = os.path.dirname(folderName)
        if not os.path.exists(path_name):
            try:
                os.makedirs(os.path.dirname(folderName))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    def write_Exp(self,file, filename, data, numIter, write=False, withArchive=True):
        if (self.typeExp == 0 or self.typeExp == 1):  # simulation expirement. save the veribles in the main
            if (file=="Xarchive"):
                self.Xarchive = data
                if (withArchive and (numIter == 0 or len(self.surr_Xarchive) == 0)):
                    self.surr_Xarchive = data
                elif withArchive:
                    self.surr_Xarchive = np.hstack((self.surr_Xarchive, data))
                elif(self.sorrugateTyp == "KRIGING"): # in kriging we update the model and not building from scratch every time (so we clean the list to update).
                    self.surr_Xarchive = []
            elif (file == 'Farchive'):
                self.Farchive = data
                if (withArchive and (numIter == 0 or len(self.surr_Xarchive) == 0)):
                    self.surr_Farchive = data
                elif withArchive:
                    self.surr_Farchive = np.hstack((self.surr_Farchive, data))
                elif (self.sorrugateTyp == "KRIGING"):  # in kriging we update the model and not building from scratch every time (so we clean the list to update).
                    self.surr_Farchive = []
            else:
                self.Sarchive = data
        #if (np.mod(numIter, float(niter / 40)) == 0 or write or numIter % 1 == 0):
        if (numIter %2 == 0 and file!="Sarchive"):
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




def main():
    problem_list = ["LABS", "Ising1D","Ising2DSquare","IsingBinaryTree","IsingTriangle","MIS","NQueens"]    #"LABS", "Ising1D","Ising2DSquare","IsingBinaryTree","IsingTriangle","MIS","NQueens"             "NKL","QAP","twoMax",
    sorrugate_list = ["empty"]   # RBFN KRIGING "RF","SVM","KNN","empty"
    exp_conditions = ["ES","HLT"]  # "ES" ,"FFD", "LHS","DOP", "HLT","Surrogate"
    conditions_str = "_".join(exp_conditions)
    DOE_popsizes = [0,50,100]#0,10,20,30,40,50,60,70,80,90
    dimension_list = [25,64,100]
    DOE_popsize_prec=[]
    sub_folder="/Result3/"
    for problem in problem_list:
        fitness_list = []
        best_vectors_list = []
        best_fitness_list = []
        num_runs_with_best_fitness_list = []
        median_list = []

        for sorrugate in sorrugate_list:
            for dim in dimension_list:
                niter = 210
                withThreads = False
                numThreads = 2
                intiNumIterWituoutSurr = 20
                isKmeans = True

                if not withThreads:
                    numThreads = 1
                FileNumToSave = 0
                fileLoc_problem = "/home/naamah/Documents/CatES/result_All/"+sub_folder+problem+"/"
                fileName = conditions_str.replace("Surrogate",sorrugate)

                while os.path.exists(fileLoc_problem+fileName):
                    FileNumToSave=FileNumToSave+1
                    fileName = conditions_str.replace("Surrogate",sorrugate) + "_" + str(FileNumToSave)

                fileLoc_general = fileLoc_problem + fileName
                fileLoc_general_IOHProfiler = fileLoc_general + "/IOHProfiler/"

                typeExp, numPar, popSize, fitness, DEVector, max_attainable, opt, fixNumber , xor_offset = init_problem(problem,dim)


                # call_F_function_counter = popSize * 70  # 2100
                call_F_function_counter = 2100
                exp_conditions, numToCompare = add_exp_conditions(exp_conditions, DOE_popsizes)

                ans = [[] for i in range(numToCompare)]

                for seed in range(numToCompare):
                    exp_condition=exp_conditions[seed]
                    start_time = datetime.now()
                    psutil.cpu_percent(interval=None)
                    psutil.cpu_times_percent(interval=None)
                    avrage = 0
                    DOE_popsize=0


                    if exp_condition == "Surrogate":
                        withSurr = True
                    else:
                        withSurr = False

                    if (exp_condition[:3] == "LHS" or exp_condition[:3] == "FFD" or exp_condition[:3] == "DOP" or exp_condition[:3] ==  "HLT" ):
                        DOE_popsize = int(int(exp_condition[4:])*call_F_function_counter/100)
                        DOE_popsize_prec = exp_condition[4:]
                        exp_condition=exp_condition[:3]

                    cat = CatES(fitness, DEVector, typeExp, numPar, popSize, seed, max_attainable, niter, opt, fixNumber,
                                sorrugate,call_F_function_counter, withSurr, problem, intiNumIterWituoutSurr, isKmeans,
                                exp_condition, DOE_popsize,fileLoc_general)

                    # cat.set_exp()
                    print("Experiment Number {}".format(seed + 1))

                    fmax = []
                    fitness_max_sum = 0
                    num_call_F = call_F_function_counter


                    if withThreads:
                        fmax.append(Parallel(n_jobs=-1, backend="multiprocessing", verbose=0)(delayed(cat.test_exp)(i) for i in range(1,numThreads+1)))

                        for i in range(numThreads):
                            fitness_list.append(fmax[0][i][0][-1])
                            best_vectors_list.append(fmax[0][i][1][-1])
                            fitness_max_sum = fitness_max_sum + fmax[0][i][0][-1]
                        num_call_F = num_call_F - fmax[0][0][2]
                        numIter = fmax[0][0][3]

                    else:
                        fmax.append(cat.test_exp(numThreads))
                        fitness_list.append(fmax[0][0][-1])
                        best_vectors_list.append(fmax[0][1][-1])
                        fitness_max_sum = fitness_max_sum + fmax[0][0][-1]
                        num_call_F = num_call_F - fmax[0][2]
                        numIter = fmax[0][3]

                    avrage, bestFitt, bestVector,ans,median,num_runs_with_max_fitness = calc_fitness_summary(avrage,fitness_max_sum,fitness_list,best_vectors_list, ans, seed,opt, numThreads)
                    write_run_general_info(fileLoc_general,numPar,exp_condition,start_time,numThreads,numIter,num_call_F,DOE_popsize_prec,intiNumIterWituoutSurr,sorrugate,isKmeans,avrage,bestFitt,bestVector,median, num_runs_with_max_fitness)
                    write_IOHProfiler_file(numThreads,fileLoc_general_IOHProfiler,seed, exp_condition,exp_conditions[seed],numPar,problem,call_F_function_counter,fitness_list)

                    median_list.append(median)
                    num_runs_with_best_fitness_list.append(num_runs_with_max_fitness)
                    best_fitness_list.append(bestFitt)

                save_to_zip_IOHProfiler(fileLoc_general_IOHProfiler,fileLoc_general)
                stats_result = write_sig_stats(numToCompare, ans,fileLoc_general)
                creat_boxPlot(numToCompare,problem, stats_result,ans,exp_conditions,fileLoc_general,max_attainable, dim,conditions_str.replace("Surrogate",sorrugate))
            write_IOHProfiler_in_one_file(fileLoc_problem,conditions_str.replace("Surrogate",sorrugate))
        write_latex_format(problem,fileLoc_problem,best_fitness_list,median_list,num_runs_with_best_fitness_list,sorrugate_list, dimension_list)

## helper functions##


def write_latex_format(problem,fileLoc_problem,max_list,median_list,num_runs_with_best_fitness_list,surrogate_list, dimension_list):
    dst=fileLoc_problem + "/Latex.txt"
    index=0
    while os.path.exists(dst):
        dst=fileLoc_problem + "/Latex"+str(index)+".txt"
        index=index+1

    exp=[]
    for surrogate in surrogate_list:
        for dimension in dimension_list:
            exp.append[surrogate+"_"+dimension]
    
    with open(dst,'a') as write_file:
        write_file.write("{}".format(problem))
        for index in range(len(median_list)):
            write_file.write(" &{} &{} & {} & {} ".format(exp[index],median_list[index],max_list[index],num_runs_with_best_fitness_list[index]))
            if index!=len(median_list)-1:
                write_file.write("&")
            else:
                write_file.write("\\\\")


def init_problem(problem, dim):
    probInit = InitProblem(dim)

    if problem == "LABS":
        return(probInit.LABS())
    elif problem == "QAP":
        return (probInit.QAP())
    elif problem == "Ising2DSquare":
        return (probInit.Ising2DSquare())
    elif problem == "IsingTriangle":
        return (probInit.IsingTriangle())
    elif problem == "Ising1D":
        return (probInit.Ising1D())
    elif problem == "IsingBinaryTree":
        return (probInit.IsingBinaryTree())
    elif problem == "MIS":
        return (probInit.MIS())
    elif problem == "NQueens":
        return (probInit.NQueens())
    elif problem == "twoMax":
        return (probInit.twoMax())
    elif problem == "NKL":
        return (probInit.NKLandscape())
    else:
        print("Undefined problem")
        exit()

def write_run_general_info(fileLoc_general,numPar,exp_condition,start_time,numThreads,numIter,num_call_F,DOE_popsize_prec,intiNumIterWituoutSurr,sorrugate,isKmeans,avrage,bestFitt,bestVector,median,num_runs_with_max_fitness):

    with open(fileLoc_general + "/General_Info.txt",
              'a') as file:
        if (exp_condition == "LHS" or exp_condition == "FFD" or exp_condition == "DOP"  or exp_condition == "HLT"):
            file.write(exp_condition + "_" + DOE_popsize_prec + "\n")
        else:
            file.write(exp_condition + "\n")

        end_time = datetime.now()
        stri = 'Duration: {}'.format(end_time - start_time)
        cpu_percent = "cpu percent: {}".format(str(psutil.cpu_percent(interval=None)))
        cpu_time = "cpu time: {}".format(str(psutil.cpu_times_percent(interval=None)))

        file.write("Dim: {}".format(numPar) + "\n")
        file.write("Num of Threads: {}".format(numThreads) + "\n")
        file.write("Num of Iterations (for each Thread): {}".format(numIter) + "\n")
        file.write("Num of calls to the F functions (for each Thread): {}".format(num_call_F) + "\n")
        file.write("Num of iterations before using the surrogate model (for each Thread): {}".format(
            intiNumIterWituoutSurr) + "\n")
        if sorrugate == "RBFN":
            file.write(
                "The centers were selected with the help K-mean algorithm: {}".format(isKmeans) + "\n")

        file.write(stri + "\n")
        file.write(cpu_percent + "\n")
        # file.write(cpu_time + "\n")
        file.write("Average Fitness: {}".format(avrage / numThreads) + "\n")
        file.write("Median Fitness: {}".format(median) + "\n")
        file.write("Number runs with Best Fitness: {}".format(num_runs_with_max_fitness) + "\n")
        file.write("Best Fitness: {}".format(bestFitt) + "\n")
        file.write("Best Vector: {}".format(str(bestVector)) + "\n\n\n")


        ###print(stri)
        ###print(cpu_percent)
    file.close()

def write_sig_stats(numToCompare, ans,fileLoc_general):
    if numToCompare>1:
        tmp_ans = [np.asarray(t[0]) for t in ans]
        stats_result = stats.f_oneway(*tmp_ans)
        # tmp_ans = np.asarray([np.asarray(t[0][0]) for t in ans])
        # stats_result = stats.f_oneway(tmp_ans)

        # significate
        print("\n\n***  Stats  ***")
        print("t: {} \np: {}".format(stats_result[0], stats_result[1]))
        with open(fileLoc_general + "/General_Info.txt",'a') as file:
            file.write("***  Stats  ***\n")
            file.write("T test: {}".format(stats_result[0]) + "\n")
            file.write("P value: {}".format(stats_result[1]) + "\n")

        return stats_result
    else:
        str ="Only one run - Can't calculate stats"
        print("\n\n***  Stats  ***")
        print(str)
        with open(fileLoc_general + "/General_Info.txt",'a') as file:
            file.write("***  Stats  ***\n")
            file.write(str+ "\n")
        return []


def write_IOHProfiler_in_one_file(fileLoc_problem, fileLoc_surrogate):
    dst = fileLoc_problem + "IOHProfiler_" + fileLoc_surrogate
    if not os.path.exists(dst):
        os.makedirs(dst)
    dirs = glob.glob(fileLoc_problem + "/" + fileLoc_surrogate + "*/IOHProfiler/")

    for dir in dirs:
        for file in os.listdir(dir):
            dst_file = dst + "/" + file
            source_file = dir + file

            if os.path.isdir(source_file):
                if not os.path.exists(dst_file):
                    shutil.copytree(source_file, dst_file)
                else:
                    for sub_file in os.listdir(source_file):
                        shutil.copy(source_file + "/" + sub_file, dst_file + "/" + sub_file)

            else:
                if not os.path.exists(dst_file):
                    shutil.copy(dir + file, dst_file)
                else:
                    with open(source_file) as f:
                        lines = f.readlines()
                        with open(dst_file, "a") as f1:
                            f1.writelines("\n")
                            f1.writelines(lines)
    shutil.make_archive(dst, 'zip', fileLoc_problem,"IOHProfiler_"+fileLoc_surrogate)


def write_IOHProfiler_file(numThreads,fileLoc_general_IOHProfiler,seed, exp_condition,exp_all_information,numPar,problem,call_F_function_counter,fitness_list):
    tempfiles = []
    creat_folder(fileLoc_general_IOHProfiler + "data_f"+str(seed + 1)+"/")

    for ind in range(numThreads):
        file_name = fileLoc_general_IOHProfiler + "exp_num_" + str(seed + 1) + "_" + exp_condition + "/profiler_format_T" + str(ind + 1) + ".dat"
        tempfiles.append(file_name)

    with open(fileLoc_general_IOHProfiler + "data_f" + str(seed + 1) + "/IOHprofiler_f" + str(seed + 1) + "_DIM" + str(numPar) + "_i" + str(numThreads) + ".dat", 'a') as file:
        for tempfile in tempfiles:
            with open(tempfile, 'r') as subfile:
                file.write(subfile.read())

    with open(fileLoc_general_IOHProfiler + "IOHprofiler_f" + str(seed + 1) + "_i" + str(numThreads) + ".info",'a') as file:
        algInfo = "WithOut Surrogate"
        if (problem == "Surrogate"):
            algInfo = "with surrogate: {}".format(surrogate)
        headers = "suite = '{}', funcId = {}, DIM = {}, algId = '{}', algInfo = '{}'\n% \ndata_f{}/IOHprofiler_f{}_DIM{}_i{}.dat". \
            format("PBO", 1, numPar, problem+"_"+exp_all_information, algInfo, str(seed+1), str(seed+1), str(numPar), str(numThreads))
        file.write(headers)
        for i in range(numThreads):
            file.write(", 1:" + str(call_F_function_counter) + "|" + str(fitness_list[i]))


def creat_folder(folderName):
    path_name = os.path.dirname(folderName)
    if not os.path.exists(path_name):
        try:
            os.makedirs(os.path.dirname(folderName))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def write_IOHProfiler_file_same_function(numThreads,fileLoc_general_IOHProfiler,seed, exp_condition,exp_all_information,numPar,problem,call_F_function_counter,fitness_list):

    tempfiles = []
    location_by_thread=fileLoc_general_IOHProfiler + "exp_" + str(seed + 1) + "_" + exp_condition
    location_by_thread_tmp=fileLoc_general_IOHProfiler + "exp_num_" + str(seed + 1) + "_" + exp_condition
    creat_folder(location_by_thread)
    creat_folder(location_by_thread+ "/data_f1/")

    for ind in range(numThreads):
        file_name = location_by_thread_tmp + "/profiler_format_T" + str(ind + 1) + ".dat"
        tempfiles.append(file_name)

    with open(location_by_thread+"/data_f1/IOHprofiler_f1_DIM" + str(numPar) + "_i" + str(numThreads) + ".dat", 'a') as file:
        for tempfile in tempfiles:
            with open(tempfile, 'r') as subfile:
                file.write(subfile.read())

    with open(location_by_thread+"/IOHprofiler_f1_i" + str(numThreads) + ".info",'a') as file:
        algInfo = "WithOut Surrogate"
        if (problem == "Surrogate"):
            algInfo = "with surrogate: {}".format(surrogate)
        headers = "suite = '{}', funcId = {}, DIM = {}, algId = '{}', algInfo = '{}'\n% \ndata_f{}/IOHprofiler_f{}_DIM{}_i{}.dat". \
            format("PBO", 1, numPar, problem+"_"+exp_all_information, algInfo, 1, 1, str(numPar), str(numThreads))
        file.write(headers)
        for i in range(numThreads):
            file.write(", 1:" + str(call_F_function_counter) + "|" + str(fitness_list[i]))

def creat_boxPlot_multiple_compersion(numToCompare,problem, stats_result,ans,exp_conditions,fileLoc_general,max_attainable,dim,conditions_str):
    try:
        fig, axs = plt.subplots(1, numToCompare, sharey=True, figsize=(30, 15))
        plt.suptitle("{} \n P Value: {}".format(problem,stats_result[1]),
                     fontsize=32)

        textstr = "*Max Fitness: {}\n*Dim: {}".format(max_attainable,dim)
        plt.gcf().text(0.90, 0.05, textstr, fontsize=20)

        for i in range(numToCompare):
            medianprops = dict(linewidth=3, color='r')
            axs[i].boxplot(np.asarray(ans[i][0]), medianprops=medianprops)
            axs[i].set_title(exp_conditions[i], fontsize=25)
            yticks = axs[i].get_yticks()
            tmp_yticks = [round(yt, 2) for yt in yticks]
            axs[i].set_yticklabels(tmp_yticks, fontsize=20)
            axs[i].set_xticks([])

        plt.savefig(fileLoc_general + "/BoxPlot_"+problem+"_d"+str(dim)+"_"+conditions_str)
        # plt.show()
        plt.close('all')
        plt.close(fig)
    except RuntimeWarning:
        print("i am in the warning")

def creat_boxPlot_no_compersion(numToCompare,problem,  stats_result,ans,exp_conditions,fileLoc_general):
    try:
        fig, axs = plt.subplots(math.ceil(numToCompare / 5), 5, sharey=True, figsize=(30, 20),squeeze=False)
        plt.suptitle("{}  \n P Value: {}".format(problem, stats_result[1]), fontsize=28)

        for plti in range(int(numToCompare % 5), 5):
            num_row=math.ceil(numToCompare / 5) -1
            fig.delaxes(axs[num_row][plti])

        for i in range(numToCompare):
            axs[int(i / 5), int(i % 5)].boxplot(np.asarray(ans[i][0]))
            axs[int(i / 5), int(i % 5)].set_title(exp_conditions[i], fontsize=22)
            # plt.ylim(min(min(a[0]), min(b[0])) * 0.98, max(max(a[0]), max(b[0])) * 1.02)
            y = axs[int(i / 5), int(i % 5)].get_yticks()
            axs[int(i / 5), int(i % 5)].set_yticklabels(y, fontsize=18)
            axs[int(i / 5), int(i % 5)].set_xticks([])

        plt.savefig(fileLoc_general + "/BoxPlot")
        # plt.show()
        plt.close('all')
        plt.close(fig)
    except RuntimeWarning:
        print("i am in the warning")

def creat_boxPlot(numToCompare,problem,  stats_result,ans,exp_conditions,fileLoc_general,max_attainable,dim,conditions_str):
    if numToCompare > 1:
        creat_boxPlot_multiple_compersion(numToCompare, problem,  stats_result, ans, exp_conditions,
                                          fileLoc_general,max_attainable,dim,conditions_str)
    else:
        pass
    #     creat_boxPlot_no_compersion(numToCompare, problem,  stats_result, ans, exp_conditions,
    #                                 fileLoc_general)

def save_to_zip_IOHProfiler(fileLoc_general_IOHProfiler,fileLoc_general):
    dirs = glob.glob(fileLoc_general_IOHProfiler + "exp_num_*/")
    for dir in dirs:
        shutil.rmtree(dir)
    # create zip file
    shutil.make_archive(fileLoc_general + "/IOHProfiler", 'zip', fileLoc_general, "IOHProfiler")

def save_to_zip_IOHProfiler_multiple_files(fileLoc_general_IOHProfiler,seed, exp_condition):

    dirs = glob.glob(fileLoc_general_IOHProfiler + "exp_num_*/")
    for dir in dirs:
        shutil.rmtree(dir)

    shutil.make_archive(fileLoc_general_IOHProfiler+ "exp_"+str(seed+1)+"_"+exp_condition , 'zip', fileLoc_general_IOHProfiler, "exp_"+str(seed+1)+"_"+exp_condition)

def calc_fitness_summary(avrage,fitness_max_sum,fitness_list,best_vectors_list, ans, seed,opt, numThreads):
    avrage = avrage + fitness_max_sum
    ans[seed].append(fitness_list)

    if (opt == 'min'):
        index = fitness_list.index(min(fitness_list))
        bestFitt = min(fitness_list)
    else:
        index = fitness_list.index(max(fitness_list))
        bestFitt = max(fitness_list)

    median = statistics.median(fitness_list)
    num_runs_with_max_fitness=fitness_list.count(bestFitt)
    bestVector=best_vectors_list[index]

    print("\nBest fitness is {} with vector: {}".format(bestFitt, str(bestVector)))
    return avrage, bestFitt, bestVector,ans,median,num_runs_with_max_fitness

def add_exp_conditions(exp_conditions,DOE_popsizes):
    numToCompare = len(exp_conditions)
    if "LHS" in exp_conditions:
        exp_conditions.remove("LHS")
        numToCompare = numToCompare + len(DOE_popsizes) - 1
        for DOE_pop_size in DOE_popsizes:
            exp_conditions.append("LHS_" + str(DOE_pop_size))

    if "FFD" in exp_conditions:
        exp_conditions.remove("FFD")
        numToCompare = numToCompare + len(DOE_popsizes) - 1
        for DOE_pop_size in DOE_popsizes:
            exp_conditions.append("FFD_" + str(DOE_pop_size))

    if "DOP" in exp_conditions:
        exp_conditions.remove("DOP")
        numToCompare = numToCompare + len(DOE_popsizes) - 1
        for DOE_pop_size in DOE_popsizes:
            exp_conditions.append("DOP_" + str(DOE_pop_size))

    if "HLT" in exp_conditions:
        exp_conditions.remove("HLT")
        numToCompare = numToCompare + len(DOE_popsizes) - 1
        for DOE_pop_size in DOE_popsizes:
            exp_conditions.append("HLT_" + str(DOE_pop_size))

    return exp_conditions,numToCompare

if __name__ == '__main__':
    main()
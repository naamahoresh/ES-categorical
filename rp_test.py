from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
import rpy2.robjects as robjects
# r.data('iris')
# print(r['iris'].head())
from rpy2.robjects.packages import importr
#
base = importr('base')
irace = importr('irace')
import subprocess
from rpy2 import robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
import numpy as np
import rpy2.robjects as ro
R = ro.r

def R_inegration():
    r.setwd("~/tuning/")
    parameters = irace.readParameters("parameters-acotsp.txt")
    scenario = irace.readScenario(filename = "scenario.txt", scenario = irace.defaultScenario())

    ans=irace(scenario = scenario, parameters = parameters)


def R_inegration_2():
    # command = 'Rscript'
    # path2script = '/home/naamah/Documents/R_project/irace_svm.R'
    # # args = ['11', '3', '9', '42']
    # cmd = [command, path2script]
    #
    # #subprocess.call
    # # check_output will run the command and store to result
    # try:
    #     x = subprocess.check_output(cmd, universal_newlines=True)
    # except subprocess.CalledProcessError as e:
    #     raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    #
    #
    # print('The maximum of the numbers is:', x)

    command = 'Rscript'
    # path2script = '/home/naamah/Documents/R_project/irace_SVM_py.R'
    path2script2 = '/home/naamah/Documents/R_project/Fraction_Factorial.R'

    cmd = [command, path2script2]
    args = [50, 2**4]

    # try:
    #     print(subprocess.call([command, args, path2script2]))
    #     # x = subprocess.call("Rscript /home/naamah/Documents/R_project/Fraction_Factorial.R --args 50 2**4", shell=True)
    #     # x = subprocess.check_output(cmd, universal_newlines=True) + args
    # except subprocess.CalledProcessError as e:
    #     raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))


import math
from pyDOE import *
FrF = importr('FrF2')
import dexpy.factorial
import dexpy.optimal
from dexpy.model import ModelOrder


def Frac():
    # x=np.asarray(FrF.FrF2(2**10,30))
    # x=np.asarray(FrF.FrF2(2**10,32))
    x=np.asarray(FrF.FrF2(2**9,20))
    x1=np.asarray([1,2,3,4,5])
    # x=np.asarray(FrF.FrF2(2**12,50))
    print(x)

def LHS():
    # initilization to QAP problem
    x1= np.arange(5)
    np.random.shuffle(x1)
    N=5
    DOE_popsize=10
    X_tmp = lhs(n=N,samples=DOE_popsize)
    # print(X)
    for i in range (1,N+1):
        X_tmp[X_tmp<(i/N)]=i

    X= np.zeros((N,DOE_popsize))
    X[:,0]=x1
    for l in range (1,DOE_popsize):
        x_raw=np.zeros((N))
        for j in range (0,N):
            x_raw[j]=X_tmp[j][l-1]
        X[:, l] =x_raw
    print(X)
    print(X_tmp)

def LHS2():

    N=4
    DOE_popsize=24
    # X_tmp = lhs(n=N,samples=DOE_popsize)
    X_tmp = lhs(DOE_popsize, N)
    for i in range(1, N + 1):
        X_tmp[X_tmp < (i / N)] = i

    # uni=np.unique(np.asarray(X_tmp), axis=0)
    # # print(uni.shape)
    # X_tmp = np.asarray([[1,2,3,1],[2,2,3,2]])
    X_tmp=X_tmp.transpose()
    unique= np.unique(X_tmp, axis=0)


    x1 = np.arange(N)
    np.random.shuffle(x1)
    X = np.zeros((N, DOE_popsize))
    X[:, 0] = x1
    for l in range(1, DOE_popsize):
        x_row = np.zeros((N))
        for j in range(0, N):
            x_row[j] = X_tmp[j][l - 1]
        X[:, l] = x_row

    print(X)


def d_optimal():
    # model_par = ['a', 'b', 'c', 'd']
    # model = '('+'+'.join(model_par)+')'
    model_par=[]
    num_par = 30
    run_count = 210
    for i in range(1,num_par+1):
        model_par.append(''.join(["X",str(i)]))
    model = '('+'+'.join(model_par)+')'

    a= dexpy.optimal.build_optimal(num_par, run_count=run_count,  model=model)
    # b= dexpy.factorial.build_factorial(factor_count=20,run_count=2**13)

    print(a)

d_optimal()
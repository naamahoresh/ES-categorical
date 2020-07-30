# QAP problem
# 2-opt heuristic
# author: Daniel Gribel -- daniel.gribel@uniriotec.br

from copy import copy, deepcopy
import math
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt

""" General testing """





""" QAP problem """

def cost(matching, d, f):
    total = 0
    size = len(matching)
    for i in range(0, size):
        for j in range(0, size):
            if i != j:
                x = f[matching[i], matching[j]] * d[i, j]
                total = total + x
    return total


def iteration(m, d, f):
    best_matching = []
    best_matching = copy(m)
    best_path = cost(m, d, f)
    size = len(m)
    i = 0
    # print m
    for i in range(0, size - 1):
        for j in range(i + 1, size):
            posI = m.index(i)
            posJ = m.index(j)
            m[posI] = j
            m[posJ] = i
            current_path = cost(m, d, f)
            if (current_path < best_path):
                best_matching = copy(m)
                best_path = current_path
    #print('final solution: ', best_matching)
    #print('final cost: ', best_path)


def get_minor(list, n):
    for i in range(0, n):
        if i not in list:
            return i
    return None


def read_file():
    ins = open("/home/naamah/Documents/CatES/nug12.dat", "r")
    i = 0
    br = 0
    rowdistance=0
    rowdflow=0
    tmpdistance=np.empty((20,20))
    tmpflow=np.empty((20,20))

    for line in ins:
        if i == 0:
            n = line.split("\n")[0]
            tmpdistance = np.empty((int(n), int(n)))
            tmpflow = np.empty((int(n), int(n)))
        elif line == "\n":
            br += 1
        else:
            if br == 1:
                tmpList=line.split(" ")
                tmpList=tmpList[0:len(tmpList)]
                numbers = [ int(x) for x in tmpList]
                tmpdistance[rowdistance]=np.asarray(numbers)
                rowdistance=rowdistance+1

            if br == 2:
                indexFlow = 0
                for k in re.split(' +', line):
                     if k != '':
                        tmpflow[rowdflow][indexFlow]= int(k.split("\n")[0])
                        indexFlow=indexFlow+1
                rowdflow=rowdflow+1
        i += 1

    return tmpdistance, tmpflow

# greedy algorithm to generate initial solution
def initial_solution(distance, flow):
    n = int(math.sqrt(len(distance)))
    best_cost_final = 0
    m_x = []

    for q in range(0, n):
        m = [q]
        m2 = [q]
        j = 0
        while j < n - 1:
            k = get_minor(m, n)
            m.append(k)
            m2 = copy(m)
            best_cost = cost(m, distance, flow)
            if j == 0:
                best_cost_final = best_cost
            for i in range(0, n):
                if i not in m:
                    m2[len(m2) - 1] = i
                    c = cost(m2, distance, flow)
                    if c < best_cost:
                        m[len(m) - 1] = i
                        best_cost = c
            j = j + 1

        if best_cost < best_cost_final:
            best_cost_final = best_cost
            m_final = copy(m)

        m_x.append(m)

    # check the best greedy initial solution, considering solutions starting with 0, 1, 2 .. n
    b_array = copy(m_x[0])
    b_cost = cost(b_array, distance, flow)

    for i in range(1, n):
        cus = cost(m_x[i], distance, flow)
        if cus < b_cost:
            b_cost = cus
            b_array = copy(m_x[i])

    #print('initial solution: ', b_array)
    #print('initial cost: ', b_cost)

    return b_array


def checkQAP ():
    distance = []
    flow = []
    read_file()
    #initial_matching = initial_solution(distance, flow)

    with open("/home/naamah/Documents/CatES/init_Qap.p", "rb") as fp:
        initial_matching = pickle.load(fp)

    iteration(initial_matching, distance, flow)


def QAPFit (X):
    #github: https://github.com/danielgribel/qap/tree/master/data
    X=X.astype(int)
    distance,flow = read_file();

    sum = 0
    size = len(X)
    for i in range(0, size):
        for j in range(0, size):
            if i != j:
                a=X[i]
                b=X[j]
                x = (flow[X[i], X[j]]) * (distance[i, j])
                sum = sum + x
    ans=sum.tolist()
    return ans

""" End of QAP problem """
""" NKLandscape problem """

def NKLandscapeFit (X):
    X=X.astype(int)
    ans=[]
    with open("/home/naamah/Documents/CatES/modelFunction.p", "rb") as fp:
        model = pickle.load(fp)
    #model = NKlandscape(len(X), 5)
    tmp = np.array_str(X).replace('\n','')
    tmp=tmp[1:len(tmp)-1].replace('\n','')
    tmp="".join(tmp.split())

    ans.append(model.compFit(tmp))
    return ans


def checkNkl ():
    with open("/home/naamah/Documents/CatES/modelFunction.p", "rb") as fp:
        model = pickle.load(fp)

    sum=0
    count=0
    function=model.getFunc()
    neighbor = model.getNeigh()
    for i in range(len(function)) :
        sum=sum+max(function[i])
        count=count+1

    print(sum/count)


""" End of NKLandscape problem """
""" Monte Carlo problem """

def simpleMonteCarlo(n, lb, ub, evals,typeExp, func=lambda x: x.dot(x),typeOpt='max') :
    history = []
    if (typeExp==1):
        xmin = np.random.randint(2, size=n)
    else:
        xmin = np.arange(n)
        np.random.shuffle(xmin)
    fopt = func(xmin)
    history.append(fopt)
    for i in range(evals) :
        if (typeExp == 1):
            x = np.random.randint(2, size=n)
        else:
            x=np.arange(n)
            np.random.shuffle(x)
        f_x = func(x)
        if (typeOpt=='max'):
            if f_x > fopt :
                xmin = x
                fopt = f_x
        else:
            if f_x < fopt :
                xmin = x
                fopt = f_x
        history.append(fopt)
        if ((i+1)%500==0):
            print("iteration: {} fmax: {}".format(i+1, fopt))
    return xmin,fopt,history

def montaCarlo(typeExp,fitness, numDim,typeOpt):
    lb,ub = 0,19
    n=numDim
    evals=100000
    xmin,fopt,history = simpleMonteCarlo(n,lb,ub,evals,typeExp,fitness,typeOpt)
    plt.semilogy(history)
    plt.show()

"""End of Monte Carlo problem """

# import pyKriging
# from pyKriging.krige import kriging
# from pyKriging.samplingplan import samplingplan

#

from joblib import Parallel, delayed
from pyDOE import *
import random
from rpy2.robjects.packages import importr
lhs_R = importr('lhs')

def initbinary(N, popSize,DE):
    ans = np.zeros((N, popSize))
    for i in range(popSize):  # initialize for each parameter in each sample its primary (random) value
        for j in range(N):
            ans[j][i] = int(math.ceil(DE[j] * random.random()) - 1)
    return ans

import dexpy.factorial

def checkLHS(N, popSize,population):
    lsh_obj = lhs(5)


def rand(popSize, N, h):
    ans = np.zeros((N, popSize))
    for i in range(0, popSize):  # initialize for each parameter in each sample its primary (random) value
        for j in range(N):
            ans[j][i] = int(math.ceil(2 * random.random()) - 1)
    return ans

def rand2(DOE_popsize,N,h):
    random.SystemRandom(random.seed())
    X_tmp = lhs(DOE_popsize, N)
    for i in range(1, N + 1):
        X_tmp[X_tmp < (i / N)] = i

    x1 = np.arange(N)+1
    random.shuffle(x1)
    X = np.zeros((N, DOE_popsize))
    X[:, 0] = np.asarray(x1)
    for l in range(1, DOE_popsize):
        x_raw = np.zeros((N))
        for j in range(0, N):
            x_raw[j] = X_tmp[j][l - 1]
        X[:, l] = x_raw

    X = X - 1
    return (X)

def rand3(DOE_popsize, N, h):
    # random.SystemRandom(random.seed())
    # y = random.Random()
    # y.jumpahead(1)
    # X_tmp = np.asarray(lhs_R.randomLHS(N,DOE_popsize))
    np.random.seed()
    X_tmp2 = lhs(DOE_popsize, N)
    # print(X_tmp2)
    return (X_tmp2)



# if __name__ == "__main__" :
    # y=[]
    # y.append((Parallel(n_jobs=-1,backend="multiprocessing", verbose=0)(delayed(rand3)(3,3,i) for i in range(4))))
    # print(y)
    # N = 20
    # pop_size = 30
    # DEVector = [0, 2]
    # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    #
    # n_neighbors = np.arange(1,10)
    # leaf_size=np.arange(10,100)
    # algorithm=np.array(["auto", "ball_tree", "kd_tree", "brute"])
    # p=np.arange(1,3)
    # samples=3
    #
    # lhs_result = lhs(6, samples=samples)
    # # print(p[1])
    # for i in range(samples):
    #     nbrs = NearestNeighbors(n_neighbors=n_neighbors[int( lhs_result[i,0] * n_neighbors.size)], leaf_size=leaf_size[int( lhs_result[i,0] * leaf_size.size)] ,
    #                             algorithm=algorithm[int( lhs_result[i,0] * algorithm.size)],p=p[int( lhs_result[i,0] * p.size)]).fit(X)

    # print(nbrs)

    # samples=3
    # lhs_result = lhs(4, samples=samples) #where n is the number of dimensions, samples as the total number of the sample space.
    # # lhs_result = np.round(lhs_result)

    # N= 4
    # DOE_popsize = 30
    # # print("lhs_result: {}".format(lhs_result))
    # X = lhs(N, DOE_popsize)
    # for i in range (1,N+1):
    #     X[X<(i/N)]=i
        # X = X.transpose()
    # import pyDOE2
    #
    # levels=[]
    # for i in range (20):
    #     levels.append(2)  # Three factors with 2, 3 or 4 levels respectively.
    # reduction = 3  # Reduce the number of experiment to approximately a third.

    # print(pbdesign(40))
    # print(pyDOE2.gsd(levels, reduction))

    # print(X)
    # level = np.asarray([2, 3])
    # level.astype(float)
    #
    # print(dexpy.factorial.build_factorial(21, 2**9))
    # fullfact([2, 4, 3])
    #checkQAP()
    #checkNkl()
    #
    # #1-NKLandscapeFit-typeOpt='max' ,2-QAPFit-typeOpt='min'
    #montaCarlo(typeExp=1,fitness=NKLandscapeFit,numDim=20, typeOpt='max')
    #
    #
    # # ans25=np.asarray([5,11,20,15,22,2,25,8,9,1,18,16,3,6,19,24,21,14,7,10,17,12,4,23,13])
    # # print("best vector for nug25: {}".format(ans25-1))
    # # ans30=np.asarray([5,12,6,13,2,21,26,24,10,9,29,28,17,1,8,7,19,25,23,22,11,16,30,4,15,18,27,3,14,20])
    # # print("best vector for nug30: {}".format(ans30-1))
    #
    #
    # sp = samplingplan(2)
    # X = sp.optimallhc(20)
    #
    # # Next, we define the problem we would like to solve
    # testfun = pyKriging.testfunctions().branin
    # y = testfun(X)
    #
    # # Now that we have our initial data, we can create an instance of a Kriging model
    # k = kriging(X, y, testfunction=testfun, name='simple')
    # k.train()
    #
    # # Now, five infill points are added. Note that the model is re-trained after each point is added
    # numiter = 5
    # for i in range(numiter):
    #     print
    #     'Infill iteration {0} of {1}....'.format(i + 1, numiter)
    #     newpoints = k.infill(1)
    #     for point in newpoints:
    #         k.addPoint(point, testfun(point)[0])
    #     k.train()
    #
    # # And plot the results
    # k.plot()
    #x=np.asarray([4,8,0,7,11,10,1,5,2,9,6,3])
    # x = np.asarray([12, 7, 9, 3, 4, 8, 11, 1, 5, 6, 10, 2])
    # x=x-1
    # print(QAPFit(x))


def IsingBinaryTreeFitOld(X): #https://www.ida.liu.se/opendsa/OpenDSA/Books/OpenDSA/html/CompleteTree.html#CompleteFIB
    X[X == 0] = -1
    X = X.astype(int)
    fitness= np.zeros(len(X[0]))
    for i in range (len(X[0])):
        spin_array =X[:,i].astype(int)
        sum_energy=0
        lattice_size=len(spin_array)
        for x in range(lattice_size):
            Parent, leftChild, rightChild = 0,0,0
            neigboord = []
            if (x!=0):
                Parent = spin_array[int((x-1)/2)]
            if (2*x + 1<lattice_size):
                leftChild = spin_array[2*x + 1]
                if (2 * x + 2 < lattice_size):
                    rightChild = spin_array[2 * x + 2]
            sum_neighbors= Parent+ leftChild+rightChild
            sum_energy += spin_array[x] * sum_neighbors

        fitness[i]=sum_energy
    return fitness

def Ising2DSquareFitOld(X):
    X[X == 0] = -1
    X = X.astype(int)
    fitness= np.zeros(len(X[0]))
    for i in range (len(X[0])):
        lattice_size=int(math.sqrt(len(X)))
        spin_array =X[:,i].astype(int)
        spin_array=spin_array.reshape((lattice_size,lattice_size))
        sum_energy=0
        for x in range(lattice_size):
            for y in range (lattice_size):
                sum_neighbors= spin_array[x, y - 1] + spin_array[x, (y + 1) % lattice_size] +\
                               spin_array[x - 1, y] + spin_array[(x + 1) % lattice_size, y]
                sum_energy += spin_array[x, y] * sum_neighbors

        fitness[i]=sum_energy
    return fitness

def Ising1DFitOld(X):
    X[X == 0] = -1
    X = X.astype(int)
    fitness= np.zeros(len(X[0]))
    for i in range (len(X[0])):
        spin_array =X[:,i].astype(int)
        sum_energy=0
        lattice_size=len(spin_array)
        for x in range(lattice_size):
            sum_neighbors= spin_array[(x+1)% lattice_size] + spin_array[(x -1) % lattice_size]
            sum_energy +=  spin_array[x] * sum_neighbors
        fitness[i]=sum_energy
    return fitness


def IsingTriangleFitOld(X):
    X[X == 0] = -1
    X = X.astype(int)
    fitness = np.zeros(len(X[0]))
    for i in range(len(X[0])):
        lattice_size = int(math.sqrt(len(X)))
        spin_array = X[:, i].astype(int)
        spin_array = spin_array.reshape((lattice_size, lattice_size))
        sum_energy = 0
        for x in range(lattice_size):
            for y in range(lattice_size):
                sum_neighbors = spin_array[(x - 1) % lattice_size, y] + \
                                spin_array[(x + 1) % lattice_size, y] + \
                                spin_array[x , (y - 1) % lattice_size] + \
                                spin_array[x, (x + 1) % lattice_size] + \
                                spin_array[(x - 1) % lattice_size, (y - 1) % lattice_size] + \
                                spin_array[(x + 1) % lattice_size, (y + 1) % lattice_size]
                sum_energy += spin_array[x, y] * sum_neighbors

        fitness[i] = sum_energy
    return fitness

def IsingBinaryTreeFit(X): #https://www.ida.liu.se/opendsa/OpenDSA/Books/OpenDSA/html/CompleteTree.html#CompleteFIB
    # X[X == 0] = -1
    X = X.astype(int)
    fitness= np.zeros(len(X[0]))
    for i in range (len(X[0])):
        spin_array =X[:,i].astype(int)
        sum_energy=0
        lattice_size=len(spin_array)
        for x in range(lattice_size):
            Parent, leftChild, rightChild = 0,0,0
            neigboord = []
            if (x!=0):
                neigboord.append(spin_array[int((x-1)/2)])
            if (2*x + 1<lattice_size):
                neigboord.append(spin_array[2*x + 1])
                if (2 * x + 2 < lattice_size):
                    neigboord.append(spin_array[2 * x + 2])
            # sum_neighbors= Parent+ leftChild+rightChild
            # sum_energy += spin_array[x] * sum_neighbors
            for neig in neigboord:
                sum_energy += (spin_array[x] * neig) + ((1 - spin_array[x])*(1 - neig))

        fitness[i]=sum_energy
    return fitness

def Ising2DSquareFit(X):
    # X[X == 0] = -1
    X = X.astype(int)
    fitness= np.zeros(len(X[0]))
    for i in range (len(X[0])):
        lattice_size=int(math.sqrt(len(X)))
        spin_array =X[:,i].astype(int)
        spin_array=spin_array.reshape((lattice_size,lattice_size))
        sum_energy=0
        for x in range(lattice_size):
            for y in range (lattice_size):
                # sum_neighbors= spin_array[x, y - 1] + spin_array[x, (y + 1) % lattice_size] +\
                #                spin_array[x - 1, y] + spin_array[(x + 1) % lattice_size, y]
                # sum_energy += spin_array[x, y] * sum_neighbors
                neigboord=[spin_array[x, y - 1] , spin_array[x, (y + 1) % lattice_size] ,  spin_array[x - 1, y] , spin_array[(x + 1) % lattice_size, y]]
                for neig in neigboord:
                    sum_energy += (spin_array[x, y] * neig) + ((1 -spin_array[x, y])*(1 - neig))

        fitness[i]=sum_energy
    return fitness

def Ising1DFit(X):
    # X[X == 0] = -1
    X = X.astype(int)
    fitness= np.zeros(len(X[0]))
    for i in range (len(X[0])):
        spin_array =X[:,i].astype(int)
        sum_energy=0
        lattice_size=len(spin_array)
        for x in range(lattice_size):
            # sum_neighbors= spin_array[(x+1)% lattice_size] + spin_array[(x -1) % lattice_size]
            # sum_energy +=  spin_array[x] * sum_neighbors
            neigboord = [spin_array[(x+1)% lattice_size] , spin_array[(x -1) % lattice_size]]
            for neig in neigboord:
                sum_energy += (spin_array[x] * neig) + ((1 - spin_array[x])*(1 - neig))
        fitness[i]=sum_energy
    return fitness


def IsingTriangleFit(X):
    # X[X == 0] = -1
    X = X.astype(int)
    fitness = np.zeros(len(X[0]))
    for i in range(len(X[0])):
        lattice_size = int(math.sqrt(len(X)))
        spin_array = X[:, i].astype(int)
        spin_array = spin_array.reshape((lattice_size, lattice_size))
        sum_energy = 0
        for x in range(lattice_size):
            for y in range(lattice_size):
                # sum_neighbors = spin_array[(x - 1) % lattice_size, y] + \
                #                 spin_array[(x + 1) % lattice_size, y] + \
                #                 spin_array[x , (y - 1) % lattice_size] + \
                #                 spin_array[x, (x + 1) % lattice_size] + \
                #                 spin_array[(x - 1) % lattice_size, (y - 1) % lattice_size] + \
                #                 spin_array[(x + 1) % lattice_size, (y + 1) % lattice_size]
                # sum_energy += spin_array[x, y] * sum_neighbors
                neigboord = [spin_array[(x - 1) % lattice_size, y] ,
                                spin_array[(x + 1) % lattice_size, y] ,
                                spin_array[x , (y - 1) % lattice_size] ,
                                spin_array[x, (x + 1) % lattice_size] ,
                                spin_array[(x - 1) % lattice_size, (y - 1) % lattice_size] ,
                                spin_array[(x + 1) % lattice_size, (y + 1) % lattice_size]]
                for neig in neigboord:
                    sum_energy += (spin_array[x, y] * neig) + ((1 -spin_array[x, y])*(1 - neig))
        fitness[i] = sum_energy
    return fitness



def N_queens_Fit(X):
    X = X.astype(int)
    fitness = np.zeros(len(X[0]))
    number_of_variables= len((X))
    N_queens = int(math.sqrt(number_of_variables))
    C = N_queens

    for ind in range(len(X[0])):
        x=X[:, ind].astype(int)
        number_of_queens_on_board = 0
        k_penalty, l_penalty, raws_penalty, columns_penalty = 0, 0, 0, 0

        for index in range (number_of_variables):
            if (x[index]==1):
                number_of_queens_on_board+=1

        for j in range (1,N_queens+1):
            sum_column = 0
            for i in range (1,N_queens+1):
                indx=((i-1)*N_queens) + ((j-1)%N_queens)
                sum_column+=x[indx]
            columns_penalty+=max(0, (-1.0+sum_column))

        for i in range(1, N_queens + 1):
            sum_k,sum_l, sum_raw = 0,0,0
            for j in range(1, N_queens + 1):
                indx=((i-1)*N_queens) + ((j-1)%N_queens)
                sum_raw+=x[indx]
            raws_penalty+=max(0.0, (-1.0+sum_raw))

        for k in range (2-N_queens,N_queens-2+1):
            sum_k=0
            for i in range(1, N_queens + 1):
                if (k+i>=1 and k+i<=N_queens):
                    indx=((i-1)*N_queens) + ((k+i-1)%N_queens)
                    sum_k += x[indx]
            k_penalty+=max(0.0, (-1.0+sum_k))

        for l in range (3,2*N_queens-1+1):
            sum_l=0
            for i in range(1, N_queens + 1):
                if (l-i>=1 and l-i<=N_queens):
                    indx=((i-1)*N_queens) + ((l-i-1)%N_queens)
                    sum_l += x[indx]
            l_penalty+=max(0.0, (-1.0+sum_l))

        fitness[ind] = number_of_queens_on_board - (C*(raws_penalty + columns_penalty+ k_penalty+l_penalty))
    return fitness


def isEdge( i, j, problem_size):
    if (i != problem_size / 2 and j == i + 1):
        return 1
    elif (i <= (problem_size / 2) - 1 and j == i + (problem_size / 2) + 1):
        return 1
    elif (i <= (problem_size / 2) and i >= 2 and j == i+(problem_size / 2)-1):
        return 1
    else:
        return 0



def MISFit(X):
    X = X.astype(int)
    fitness = np.zeros(len(X[0]))
    number_of_variables= len((X))
    number_of_variables_even = number_of_variables

    if (number_of_variables % 2 != 0):
        number_of_variables_even = number_of_variables - 1

    for ind in range(len(X[0])):
        x=X[:, ind].astype(int)
        num_of_ones, sum_edges_in_the_set = 0, 0
        ones_array=[]

        for index in range (number_of_variables_even):
            if (x[index] == 1):
                ones_array.append(index)
                num_of_ones += 1

        for i in range (num_of_ones):
            for j in range (i+1,num_of_ones):
                if (isEdge(ones_array[i]+1, ones_array[j]+1, number_of_variables_even) == 1):
                    sum_edges_in_the_set += 1

        fitness[ind] = num_of_ones - (number_of_variables_even * sum_edges_in_the_set)
        print("num_of_ones {}".format(num_of_ones))
        print("sum_edges_in_the_set {}".format(sum_edges_in_the_set))
        print("number_of_variables_even {}".format(number_of_variables_even))
    return fitness


X=np.ones((10,2))

# X=np.asarray([1,  0,  1,  0,  1,  1,  0,  0, 1,  1])
# # # X=X.reshape((len(X),1))
# X=X.reshape((10,1))

# print(MISFit(X))

# print(N_queens_Fit(X))
# print(IsingBinaryTreeFit(X))
# print(Ising1DFit(X))
# print(Ising2DSquareFit(X))
# print(IsingTriangleFit(X))
# print("")
# print( IsingBinaryTreeFitOld(X))
# print(Ising1DFitOld(X))
# print(Ising2DSquareFitOld(X))
# print(IsingTriangleFitOld(X))
import pynolh

dim = 50
m, q, r = pynolh.params(dim)
conf = range(q)
remove = range(dim - int(r), dim)
nolh = pynolh.nolh(conf, remove)
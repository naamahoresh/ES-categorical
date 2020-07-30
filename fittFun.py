import numpy as np
import re
import math
import random
import pickle as pickle
import sys
sys.path.insert(0, '/home/naamah/Documents/CatES/')
from NKLandscape_class import NKlandscape



def SwedishPumpFit(X):
    """ The correlation function, assumes a numpy vector {-1,+1} as input """
    X[X == 0] = -1
    ans = []
    for i in range(len(X[0])):
        n = len(X[:, i])
        E = []
        for k in range(1, n):
            E.append((X[0:n - k, i].dot(X[k:, i])) ** 2)
        ans.append((n ** 2) / (2 * sum(E)))
    return ans


def NKLandscapeFit(X):
    X = X.astype(int)
    ans = []
    with open("/home/naamah/Documents/CatES/result_All/NKL/init_NKL.pickle", "rb") as fp:
        model = pickle.load(fp)

    #model = NKlandscape(len(X), 5)
    for x in range(len(X[0])):
        tmp = np.array_str(X[:, x]).replace('\n', '')
        tmp = tmp[1:len(tmp) - 1].replace('\n', '')
        tmp = "".join(tmp.split())

        ans.append(model.compFit(tmp))
    return ans


def QAPFit(X):
    # github: https://github.com/danielgribel/qap/tree/master/data
    X = X.astype(int)
    ans = []
    distance,flow = read_file(len(X))
    sum = 0
    size = len(X)
    for i in range(0, size):
        for j in range(0, size):
            if i != j:
                try:
                    x = flow[X[i], X[j]] * distance[i, j]
                except IndexError:
                    print("Error in QAPFit funcation")
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


# print(NKLandscapeFit(np.random.randint(2, size=(30,10))))
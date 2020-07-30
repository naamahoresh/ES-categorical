import re
import numpy as np
import pickle as pickle
import math



class InitProblem():

    def __init__(self,dim):
        self.xor_offset=np.zeros((1,1))
        self.numPar_global=int(dim)


    def get_xor_offset(self):
        return self.xor_offset

    def LABS(self):
        typeExp = 1  # lab (0) or simulation (1)
        numPar = self.numPar_global
        popSize = 30
        fitness = self.LABSFit
        DEVector = [0, 2]
        max_attainable = np.inf
        if numPar==16:
            max_attainable=16/3
        elif numPar == 20:
            max_attainable = 7.692
        elif numPar == 25:
            max_attainable = 8.681
        elif numPar==64:
            max_attainable =9.846
        opt = 'max'
        repeated_Number = True
        self.xor_offset=np.random.randint(2, size=numPar)
        return typeExp, numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number,self.xor_offset

    def NQueens(self):
        typeExp = 1  # lab (0) or simulation (1)
        numPar = int(math.sqrt(self.numPar_global))
        popSize = 30
        fitness = self.NQueensFit
        DEVector = [0, 2]
        max_attainable = numPar
        opt = 'max'
        repeated_Number = True
        self.xor_offset=np.random.randint(2,size=int(numPar*numPar))
        return typeExp, numPar*numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number,self.xor_offset

    def MIS(self):
        typeExp = 1  # lab (0) or simulation (1)
        numPar = self.numPar_global
        # if(numPar%2!=0 ):
        #     numPar=numPar-1
        popSize = 30
        fitness = self.MISFit
        DEVector = [0, 2]
        max_attainable = (numPar/2)+1
        opt = 'max'
        repeated_Number = True
        self.xor_offset=np.random.randint(2, size=numPar)
        return typeExp, numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number,self.xor_offset

    def Ising1D (self):
        typeExp = 1  # lab (0) or simulation (1)
        numPar = self.numPar_global
        popSize = 30
        fitness =self.Ising1DFit
        DEVector = [0, 2]
        max_attainable = numPar*2
        opt = 'max'
        repeated_Number = True
        self.xor_offset=np.random.randint(2, size=numPar)
        return typeExp, numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number,self.xor_offset

    def IsingBinaryTree (self):
        typeExp = 1  # lab (0) or simulation (1)
        numPar = self.numPar_global
        popSize = 30
        fitness =self.IsingBinaryTreeFit
        DEVector = [0, 2]
        max_attainable = (numPar*2)-2
        opt = 'max'
        repeated_Number = True
        self.xor_offset=np.random.randint(2, size=numPar)
        return typeExp, numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number,self.xor_offset


    def Ising2DSquare(self):
        typeExp = 1  # lab (0) or simulation (1)
        numPar = int(math.sqrt(self.numPar_global))
        popSize = 30
        fitness = self.Ising2DSquareFit
        DEVector = [0, 2]
        max_attainable = numPar*numPar*4
        opt = 'max'
        repeated_Number = True
        self.xor_offset=np.random.randint(2, size=int(numPar*numPar))
        return typeExp, numPar*numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number,self.xor_offset

    def IsingTriangle(self):#https://github.com/mattsep/ising/blob/master/src/lattice.py
        typeExp = 1  # lab (0) or simulation (1)
        numPar = int(math.sqrt(self.numPar_global))
        popSize = 30
        fitness = self.IsingTriangleFit
        DEVector = [0, 2]
        max_attainable = numPar*numPar*6
        opt = 'max'
        repeated_Number = True
        self.xor_offset=np.random.randint(2, size=int(numPar*numPar))
        return typeExp, numPar*numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number,self.xor_offset

    def maxOne(self):
        typeExp = 1  # lab (0) or simulation (1)
        numPar = 100
        popSize = 50
        fitness = self.maxOneFit
        DEVector = [0, 2]
        max_attainable = numPar
        opt = 'max'
        repeated_Number = True
        self.xor_offset=np.random.randint(2, size=numPar)
        return typeExp, numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number,self.xor_offset

    def twoMax(self):
        typeExp = 1  # lab (0) or simulation (1)
        numPar = 100
        popSize = 50
        fitness = self.twoMaxFit
        DEVector = [0, 2]
        max_attainable = 100
        opt = 'max'
        repeated_Number = True
        self.xor_offset=np.random.randint(2, size=numPar)
        return typeExp, numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number,self.xor_offset

    def QAP(self):
        typeExp = 0  # lab (0) or simulation (1)
        numPar = 12  # number of dominations / parameters
        popSize = 30  # number of sample in each generation
        fitness = self.QAPFit  # the fitness function
        DEVector = [
            1]  # number of levels for each dimnation. the index 0 will represent all the dimnation have the same lav el (0=same lavel, 1= different)
        for i in range(popSize):
            DEVector.append(numPar)
        max_attainable = np.inf
        opt = 'min'
        repeated_Number = False
        self.xor_offset=np.random.randint(2, size=numPar)
        return typeExp, numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number,self.xor_offset

    def NKLandscape(self):
        typeExp = 1  # lab (0) or simulation (1)
        numPar = 20  # number of dominations / parameters
        popSize = 30  # number of sample in each generation
        fitness = self.NKLandscapeFit  # the fitness function
        DEVector = [0,
                    2]  # number of levels for each dimnation. the index 0 will represent all the dimnation have the same lavel (0=same lavel, 1= different)
        max_attainable = np.inf
        opt = 'max'
        repeated_Number = True  # is the number in each vector in the population can repeted?
        self.xor_offset=np.random.randint(2, size=numPar)

        # initialize the NKlandscape model - only if need to change the model!!!!!!!!!!!!
        # model = NKlandscape(numPar, 2)
        # with open("/home/naamah/Documents/CatES/result_All/NKL/init_NKL.pickle", "wb") as fp:  # Pickling
        #    pickle.dump(model, fp, protocol=2)


        return typeExp, numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number,self.xor_offset

    def protein(self):
        typeExp = 1  # lab (0) or simulation (1)
        numPar = 10  # number of dominations / parameters
        popSize = 24  # number of sample in each generation
        fitness = self.tmp  # the fitness function
        DEVector = [1, 4, 5, 3, 6, 5, 4, 7, 2, 11, 3]  # number of levels for each dimnation the index 0 will represent
        # if it the same lavel (0=same lavel, 1= different)
        max_attainable = np.inf
        opt = 'max'
        repeated_Number = True
        self.xor_offset=np.random.randint(2, size=popSize)
        return typeExp, numPar, popSize, fitness, DEVector, max_attainable, opt, repeated_Number,self.xor_offset


    def read_file_QAP(self,n):
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


    def maxOneFit(self,X):
        mu = len(X[0])
        fitness = []
        for i in range(mu):
            fitness.append(sum(X[:, i]))
        return fitness


    def tmp(self,X):
        X = X + 1
        local_state = np.random.RandomState(1)
        return local_state.rand(1, 24) * 5


    def LABSFit(self,X):
        """ The correlation function, assumes a numpy vector {-1,+1} as input """
        fitness = []
        for ind in range(len(X[0])):
            # X_tmp = self.xor_transformation(X[:, ind])
            X_tmp = X[:, ind].astype(int)
            n = len(X_tmp)
            X_tmp[X_tmp == 0] = -1
            E = []
            for k in range(1, n):
                E.append((X_tmp[0:n - k].dot(X_tmp[k:])) ** 2)
                # E.append((X_tmp[0:n - k, i].dot(X_tmp[k:, i])) ** 2)

            fitness.append((float(n ** 2) / float(2 * sum(E))))
        return fitness




    def IsingBinaryTreeFit(self,X):#https://www.ida.liu.se/opendsa/OpenDSA/Books/OpenDSA/html/CompleteTree.html#CompleteFIB
        X = X.astype(int)
        fitness= np.zeros(len(X[0]))
        for ind in range (len(X[0])):
            # spin_array =self.xor_transformation(X[:,ind]).astype(int)
            spin_array= X[:, ind].astype(int)
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
                for neig in neigboord:
                    sum_energy += (spin_array[x] * neig) + ((1 - spin_array[x])*(1 - neig))

            fitness[ind]=sum_energy
        return fitness

    def Ising2DSquareFit(self,X):
        X = X.astype(int)
        fitness= np.zeros(len(X[0]))
        for ind in range (len(X[0])):
            lattice_size=int(math.sqrt(len(X)))
            # spin_array =self.xor_transformation(X[:,ind]).astype(int)
            spin_array= X[:, ind].astype(int)
            spin_array=spin_array.reshape((lattice_size,lattice_size))
            sum_energy=0
            for x in range(lattice_size):
                for y in range (lattice_size):
                    neigboord=[spin_array[x, y - 1] , spin_array[x, (y + 1) % lattice_size] , spin_array[x - 1, y] , spin_array[(x + 1) % lattice_size, y]]
                    for neig in neigboord:
                        sum_energy += (spin_array[x, y] * neig) + ((1 -spin_array[x, y])*(1 - neig))

            fitness[ind]=sum_energy
        return fitness

    def Ising1DFit(self,X):
        X = X.astype(int)
        fitness= np.zeros(len(X[0]))
        for ind in range (len(X[0])):
            # spin_array =self.xor_transformation(X[:,ind]).astype(int)
            spin_array= X[:, ind].astype(int)
            sum_energy=0
            lattice_size=len(spin_array)
            for x in range(lattice_size):
                neigboord = [spin_array[(x+1)% lattice_size] , spin_array[(x -1) % lattice_size]]
                for neig in neigboord:
                    sum_energy += (spin_array[x] * neig) + ((1 - spin_array[x])*(1 - neig))
            fitness[ind]=sum_energy
        return fitness


    def IsingTriangleFit(self,X):
        X = X.astype(int)
        fitness = np.zeros(len(X[0]))
        for ind in range(len(X[0])):
            lattice_size = int(math.sqrt(len(X)))
            # spin_array =self.xor_transformation(X[:,ind]).astype(int)
            spin_array= X[:, ind].astype(int)
            spin_array = spin_array.reshape((lattice_size, lattice_size))
            sum_energy = 0
            for x in range(lattice_size):
                for y in range(lattice_size):
                    neigboord = [spin_array[(x - 1) % lattice_size, y] ,
                                    spin_array[(x + 1) % lattice_size, y] ,
                                    spin_array[x , (y - 1) % lattice_size] ,
                                    spin_array[x, (x + 1) % lattice_size] ,
                                    spin_array[(x - 1) % lattice_size, (y - 1) % lattice_size] ,
                                    spin_array[(x + 1) % lattice_size, (y + 1) % lattice_size]]
                    for neig in neigboord:
                        sum_energy += (spin_array[x, y] * neig) + ((1 -spin_array[x, y])*(1 - neig))
            fitness[ind] = sum_energy
        return fitness


    def NQueensFit(self,X):
        fitness = np.zeros(len(X[0]))
        number_of_variables= len((X))
        N_queens = int(math.sqrt(number_of_variables))
        C = N_queens

        for ind in range(len(X[0])):
            # x = self.xor_transformation(X[:, ind]).astype(int)
            x =X[:, ind].astype(int)
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


    def isEdge( self, i, j, problem_size):
        if (i != problem_size / 2 and j == i + 1):
            return 1
        elif (i <= (problem_size / 2) - 1 and j == i + (problem_size / 2) + 1):
            return 1
        elif (i <= (problem_size / 2) and i >= 2 and j == i+(problem_size / 2)-1):
            return 1
        else:
            return 0

    def MISFit(self, X):
        X = X.astype(int)
        fitness = np.zeros(len(X[0]))
        number_of_variables= len((X))
        number_of_variables_even = number_of_variables

        if (number_of_variables % 2 != 0):
            number_of_variables_even = number_of_variables - 1

        for ind in range(len(X[0])):
            # x=self.xor_transformation(X[:, ind]).astype(int)
            x = X[:, ind].astype(int)
            num_of_ones, sum_edges_in_the_set = 0, 0
            ones_array=[]

            for index in range (number_of_variables_even):
                if (x[index] == 1):
                    ones_array.append(index)
                    num_of_ones += 1

            for i in range (num_of_ones):
                for j in range (i+1,num_of_ones):
                    if (self.isEdge(ones_array[i]+1, ones_array[j]+1, number_of_variables_even) == 1):
                        sum_edges_in_the_set += 1

            fitness[ind] = num_of_ones - (number_of_variables_even * sum_edges_in_the_set)

        return fitness

    def twoMaxFit(self, X):
        X = X.astype(int)
        fitness= np.zeros(len(X[0]))
        for ind in range (len(X[0])):
            x=self.xor_transformation(X[:, i]).astype(int)
            countOne = np.count_nonzero(x)
            # tmp= abs((len(X)/2)- countOne)+(len(X)/2)
            fitness[ind]= max(countOne, len(X)-countOne)
        return fitness

    def QAPFit(self, X):
        # github: https://github.com/danielgribel/qap/tree/master/data
        X = X.astype(int)
        distance, flow = self.read_file_QAP(len(X))
        sum = 0
        size = len(X)
        for i in range(0, size):
            for j in range(0, size):
                if i != j:
                    x = flow[X[i], X[j]] * distance[i, j]
                    sum = sum + x
        fitness = sum.tolist()
        return fitness


    def xor_transformation(self, X):
        X_tmp=np.zeros((len(X)))
        for ind in range (len(X)):
            X_tmp[ind]= (self.xor_offset[ind]!=X[ind])
        return X_tmp


  #
  # def NKLandscapeFit(self,X):
  #       X = X.astype(int)
  #       fitness = []
  #       with open("/home/naamah/Documents/CatES/result_All/NKL/init_NKL.p", "rb") as fp:
  #           model = pickle.load(fp)
  #       for x in range(len(X[0])):
  #           tmp = np.array_str(X[:, x]).replace('\n', '')
  #           tmp = tmp[1:len(tmp) - 1].replace('\n', '')
  #           tmp = "".join(tmp.split())
  #
  #           fitness.append(model.compFit(tmp))
  #       return fitness
  #
  #   def Ising2DSquareFit_old(self,X):
  #       X_tmp=np.copy(X)
  #       X_tmp[X_tmp == 0] = -1
  #       X_tmp = X_tmp.astype(int)
  #       fitness= np.zeros(len(X_tmp[0]))
  #       for i in range (len(X_tmp[0])):
  #           lattice_size=int(math.sqrt(len(X_tmp)))
  #           spin_array =X_tmp[:,i].astype(int)
  #           spin_array=spin_array.reshape((lattice_size,lattice_size))
  #           sum_energy=0
  #           for x in range(lattice_size):
  #               for y in range (lattice_size):
  #                   sum_neighbors= spin_array[x, y - 1] + spin_array[x, (y + 1) % lattice_size] +\
  #                                  spin_array[x - 1, y] + spin_array[(x + 1) % lattice_size, y]
  #                   sum_energy += 2 * spin_array[x, y] * sum_neighbors
  #
  #           fitness[i]=sum_energy
  #       return fitness
  #
  #   def Ising1DFit_old(self,X):
  #       X_tmp=np.copy(X)
  #       X_tmp[X_tmp == 0] = -1
  #       X_tmp = X_tmp.astype(int)
  #       fitness= np.zeros(len(X_tmp[0]))
  #       for i in range (len(X_tmp[0])):
  #           spin_array =X_tmp[:,i].astype(int)
  #           sum_energy=0
  #           lattice_size=len(spin_array)
  #           for x in range(lattice_size):
  #               sum_neighbors= spin_array[(x+1)% lattice_size] + spin_array[(x -1) % lattice_size]
  #               sum_energy += 2 * spin_array[x] * sum_neighbors
  #           fitness[i]=sum_energy
  #       return fitness
  #
  #
  #   def IsingTriangleFit_old(self,X):
  #       X_tmp=np.copy(X)
  #       X_tmp[X_tmp == 0] = -1
  #       X_tmp = X_tmp.astype(int)
  #       fitness= np.zeros(len(X_tmp[0]))
  #       for i in range(len(X_tmp[0])):
  #           lattice_size = int(math.sqrt(len(X_tmp)))
  #           spin_array = X_tmp[:, i].astype(int)
  #           spin_array = spin_array.reshape((lattice_size, lattice_size))
  #           sum_energy = 0
  #           for x in range(lattice_size):
  #               for y in range(lattice_size):
  #                   sum_neighbors = spin_array[(x - 1) % lattice_size, y] + \
  #                                   spin_array[(x + 1) % lattice_size, y] + \
  #                                   spin_array[x , (y - 1) % lattice_size] + \
  #                                   spin_array[x, (x + 1) % lattice_size] + \
  #                                   spin_array[(x - 1) % lattice_size, (y - 1) % lattice_size] + \
  #                                   spin_array[(x + 1) % lattice_size, (y + 1) % lattice_size]
  #                   sum_energy += 2 * spin_array[x, y] * sum_neighbors
  #
  #           fitness[i] = sum_energy
  #       return fitness
  #
  #
  #
  #
  #   def IsingBinaryTreeFit_old(self,X):#https://www.ida.liu.se/opendsa/OpenDSA/Books/OpenDSA/html/CompleteTree.html#CompleteFIB
  #       X_tmp=np.copy(X)
  #       X_tmp[X_tmp == 0] = -1
  #       X_tmp = X_tmp.astype(int)
  #       fitness= np.zeros(len(X_tmp[0]))
  #       for i in range (len(X_tmp[0])):
  #           spin_array =X_tmp[:,i].astype(int)
  #           sum_energy=0
  #           lattice_size=len(spin_array)
  #           for x in range(lattice_size):
  #               Parent, leftChild, rightChild = 0,0,0
  #               if (x!=0):
  #                   # Find parent
  #                   # parent ind = (myInd-1)\2
  #                   Parent=spin_array[int((x-1)/2)]
  #               if (2*x + 1<lattice_size):
  #
  #                   leftChild=spin_array[2*x + 1]
  #                   if (2 * x + 2 < lattice_size):
  #                       rightChild = spin_array[2 * x + 2]
  #               sum_neighbors= Parent+ leftChild+rightChild
  #               sum_energy += 2 * spin_array[x] * sum_neighbors
  #           fitness[i]=sum_energy
  #       return fitness
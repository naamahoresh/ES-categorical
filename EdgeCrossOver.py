import numpy as np
import copy

class ERO:
    '''
    ERO is a class to generate two new offsprings
    from two parents, using crossover.
    It is based on edge recombination operator algorithm
    '''
    def __init__(self, gen1, gen2, seed=None):
        self.n = len(gen1)
        self.all_vertices = set(gen1)
        self.random = np.random.RandomState(seed)
        self.adj_mat = self.build_adjacency_matrix(gen1, gen2)
        self.gen1_node = self.random.choice(gen1)
        self.gen2_node = self.random.choice(gen2)


    def build_adjacency_matrix(self, gen1, gen2):
        '''
        Build the adjancy matrix from two parents
        '''
        n = self.n

        # Creating the adjacency matrix from the two vectors
        adj_mat = {}
        for i in range(n):
            gen1_ind = list(gen1).index(i)
            gen2_ind = list(gen2).index(i)
            adj_mat[i] = set((gen1[( gen1_ind +1) % n], gen1[( gen1_ind -1) % n], gen2[( gen2_ind +1) % n], gen2[( gen2_ind -1) % n]))

        return adj_mat

    def generate_crossover(self):
        '''
        Create two new offsprings
        '''
        return self.fill_gen(self.gen1_node), self.fill_gen(self.gen2_node)

    def fill_gen(self, first_node):
        '''
        Create a new offspring
        first_node - the node from which the offspring starts it's graph
        '''
        available_vertices = set(self.all_vertices)
        adj_mat = copy.deepcopy(self.adj_mat)
        K = []
        N = first_node
        while len(K) < self.n:
            K.append(N)
            available_vertices.remove(N)
            for i in available_vertices:
                e = adj_mat[i]
                if N in e:
                    e.remove(N)

            newN = self.find_new_N(adj_mat, N, available_vertices)
            adj_mat[N] = None
            N = newN

        return K

    def find_new_N(self, adj_mat, oldN, available_vertices):
        '''
        find the next neighbor.
        adj_mat - graph containing who are the neighbors in the parents
        oldN - the neighbor we currently added
        available_vertices - the new neighbors to choose from. In case the oldN's neighbor are already in the new graph
        return: the new N
        '''
        s = adj_mat[oldN]
        n = self.n

        # All the neighbors have been used already
        if len(s) == 0:
            tmp = list(available_vertices)
            if not tmp:
                return None
            return self.random.choice(tmp)

        # Find the neighbor with the fewest neighbors
        min_neigh_list = []
        min_neigh = n
        for neigh in s:
            neigh_node = adj_mat[neigh]
            l = len(neigh_node)
            if l < min_neigh:
                min_neigh_list = [neigh]
                min_neigh = l
            elif l == min_neigh:
                min_neigh_list.append(neigh)

        # Randomly choose the neighbor
        return self.random.choice(min_neigh_list)

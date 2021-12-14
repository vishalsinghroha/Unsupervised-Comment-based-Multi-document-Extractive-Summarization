"""NSGA-II related functions"""

import functools
from filecmp import cmp
from collections import Counter
from joblib.numpy_pickle_utils import xrange
from functools import cmp_to_key
from SMEA.population import Population
import random
import numpy as np
# from clustering.clusternew import Kmeans_clu, Kmeans_EMD, kmeans_cosine


from create_children.Evoution import MatingPool_Generation, MBDE

def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

class NSGA2Utils(object):

    def __init__(self, problem, num_of_individuals):

        self.problem = problem
        self.num_of_individuals = num_of_individuals
        # self.mutation_strength = mutation_strength
        # self.number_of_genes_to_mutate = num_of_genes_to_mutate
        # self.num_of_tour_particips = num_of_tour_particips

    def fast_nondominated_sort(self, population):
        population.fronts = []
        population.fronts.append([])
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = set()
            for other_individual in population:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.add(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                population.fronts[0].append(individual)
                individual.rank = 0
        i = 0
        while len(population.fronts[i]) > 0:
            temp = []
            for individual in population.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            population.fronts.append(temp)
        del population.fronts[len(population.fronts) - 1]

    def my_new_cmp(self, a, b):
        temp1 = a > b
        temp1 = int(temp1)
        temp2 = a < b
        temp2 = int(temp2)
        result = temp1 - temp2
        return result

    def __sort_objective(self, val1, val2, m):
        return self.my_new_cmp(val1.objectives[m], val2.objectives[m])

    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            for m in range(len(front[0].objectives)):
                front = sorted(front, key=cmp_to_key(functools.partial(self.__sort_objective, m=m)))
                front[0].crowding_distance = float('inf')
                front[solutions_num - 1].crowding_distance = float('inf')
                # print("Front solution 0 and last are assigned highest fns values equal to : ",self.problem.max_objectives[m] )
                # for index, value in enumerate(front[1:solutions_num-1]):
                # front[index].crowding_distance += ((front[index+1].crowding_distance - front[index-1].crowding_distance) / (self.problem.max_objectives[m] - self.problem.min_objectives[m]))
                for ll in range(1, solutions_num - 1):
                    # print("front at index {0} and crowding distance, objective funtion value : ".format(ll), front[ll], front[ll].crowding_distance, front[ll].objectives[m])
                    # print("({0}+1) crowding distance and front : ".format(ll), front[ll + 1].objectives[m], front[ll + 1])
                    # print("({0}-1) crowding distance : ".format(ll), front[ll - 1].objectives[m], front[ll - 1])
                    # print("max and min objective : ", self.problem.max_objectives[m], self.problem.min_objectives[m])
                    # print("difference b/w crowding : ", (front[ll + 1].objectives[m] - front[ll - 1].objectives[m]))
                    # print("difference b/w max and min objective : ", (self.problem.max_objectives[m] - self.problem.min_objectives[m]))
                    a = (front[ll + 1].objectives[m] - front[ll - 1].objectives[m])
                    b = self.problem.max_objectives[m] - self.problem.min_objectives[m]
                    q = float(a) / (b + 0.001)
                    front[ll].crowding_distance = front[ll].crowding_distance + q
                    # print("calculated : ", front[ll].crowding_distance)

                # for ll in range(1,solutions_num-1):
                # front[ll].crowding_distnace=front[ll].crowding_distnace + ((front[ll+1].crowding_distance -front[ll-1].crowding_distance)/ (self.problem.max_objectives[m] - self.problem.min_objectives[m]))

    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or \
                ((individual.rank == other_individual.rank) and (
                        individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def create_initial_population(self, min_sen, max_sen, max_len_solution):
        population = Population()
        for i in range(int(self.num_of_individuals)):
            individual = self.problem.generateIndividual(min_sen, max_sen, max_len_solution, flag=0)
            self.problem.calculate_objectives(individual)
            population.population.append(individual)

        return population

    def create_children(self, min_sen, max_sen, population, H, TT_WMD_matrix,
                        MAX_reader_attention_matrix,
                        MAX_density_based_score_matrix,
                        MAX_objective4_score_matrix,
                        MAX_objective5_score_matrix,
                        MAX_TWEET_length_matrix, max_len_solution):
        children = []
        solution_no = 0
        using_som = 0
        no_using_som = 0
        for individual in population:
            temp1 = list(individual.features)
            # flag=1
            MatingPool = random.sample(xrange(0, len(population)),
                                       3)  # np.arange(0, len(population)), 1   #mating_generate.mating_pool_generation(H, L, solution_no, neuron_weight,  len(population), beta=0.8)

            mbde = MBDE(MatingPool, population, solution_no, temp1, TT_WMD_matrix,
                        MAX_reader_attention_matrix,
                        MAX_density_based_score_matrix,
                        MAX_objective4_score_matrix,
                        MAX_objective5_score_matrix,
                        MAX_TWEET_length_matrix, max_len_solution, b=6, CR=0.8, F=0.8, min_sent_in_summary=min_sen,
                        max_sent_in_summary=max_sen)
            new_solutions, number_of_ones_in_solutions = mbde.Generate()
            # print "numbe of ones :", number_of_ones_in_solutions

            """below lines are for creating child population"""
            # print "number of child solutions generated for solution {0} : ".format(solution_no), len(new_solutions)
            # if len(new_solutions)>=1:
            for p in range(len(new_solutions)):
                child1 = self.problem.generateIndividual(min_sen, max_sen, max_len_solution, flag=1,
                                                         sol=new_solutions[p], k=number_of_ones_in_solutions[p])
                self.problem.calculate_objectives(child1)
                children.append(child1)
            # print "============================="
            solution_no += 1

        # print("Total number of child solutions for current population : ", len(children))
        return children

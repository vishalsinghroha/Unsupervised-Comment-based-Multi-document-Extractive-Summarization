"""Module with definition of ZDT problem interface"""

from SMEA.individual import Individual
from SMEA.problems import Problem
import random
import functools
# from clustering.clusternew import Kmeans_clu, Kmeans_EMD
import numpy as np


class ZDT(Problem):

    def __init__(self, zdt_definitions, solution_length):
        self.zdt_definitions = zdt_definitions
        self.max_objectives = [None, None, None, None, None]
        self.min_objectives = [None, None, None, None, None]
        self.problem_type = None
        self.n = solution_length

    def __dominates(self, individual2, individual1):
        worse_than_other = self.zdt_definitions.f1(individual1) >= self.zdt_definitions.f1(individual2) \
                           and self.zdt_definitions.f2(individual1) >= self.zdt_definitions.f2(individual2) \
                           and self.zdt_definitions.f3(individual1) >= self.zdt_definitions.f3(individual2) \
                           and self.zdt_definitions.f4(individual1) >= self.zdt_definitions.f4(individual2) \
                           and self.zdt_definitions.f5(individual1) >= self.zdt_definitions.f5(individual2)

        better_than_other = self.zdt_definitions.f1(individual1) > self.zdt_definitions.f1(
            individual2) or self.zdt_definitions.f2(individual1) > self.zdt_definitions.f2(
            individual2) \
                            or self.zdt_definitions.f3(individual1) > self.zdt_definitions.f3(individual2) \
                            or self.zdt_definitions.f4(individual1) > self.zdt_definitions.f4(individual2) \
                            or self.zdt_definitions.f5(individual1) > self.zdt_definitions.f5(individual2)
        return worse_than_other and better_than_other

    def generateIndividual(self, smin, smax, max_len_solution, flag=0, sol=1, k=2):
        """
        :param smin: minimum number of sentences in the summary
        :param smax: maximum number of sentences in the summary
        :param max_len_solution:  maximum length of solution
        :param flag:
        :return: population of solutions
        """
        individual = Individual()
        if flag == 0:
            individual.features = []
            x = random.randint(smin, smax)
            one = np.ones(x)
            zero = np.zeros(max_len_solution - x)
            sol_arr = np.concatenate((one, zero))
            np.random.shuffle(sol_arr)
            individual.features = sol_arr.tolist()
            individual.K = x
            individual.dominates = functools.partial(self.__dominates, individual1=individual)
            self.calculate_objectives(individual)
            return individual
        else:
            # individual = Individual()
            individual.features = sol
            individual.K = k
            individual.dominates = functools.partial(self.__dominates, individual1=individual)
            # print("child ")
            self.calculate_objectives(individual)
            return individual

    def calculate_objectives(self, individual):
        individual.objectives = []
        individual.objectives.append(self.zdt_definitions.f1(individual))
        individual.objectives.append(self.zdt_definitions.f2(individual))
        individual.objectives.append(self.zdt_definitions.f3(individual))
        individual.objectives.append(self.zdt_definitions.f4(individual))
        individual.objectives.append(self.zdt_definitions.f5(individual))

        for i in range(5):
            if self.min_objectives[i] is None or individual.objectives[i] < self.min_objectives[i]:
                self.min_objectives[i] = individual.objectives[i]
            if self.max_objectives[i] is None or individual.objectives[i] > self.max_objectives[i]:
                self.max_objectives[i] = individual.objectives[i]

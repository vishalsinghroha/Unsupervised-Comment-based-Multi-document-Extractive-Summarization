from random import uniform
import numpy as np
import random
def Repair_Solution(y, x_count, b,a):
    '''
    
    :param map_matrix: Mapped matrix(neuron matrix after one to one mapping)
    :param y: Trial solution generated
    :param x: population
    :param x_count: main features of current solution y
    :return: Repaired solution of y
    '''
    yRep = []

    actual_length_y= len(y)

    for k in range(x_count):
            if y[k] < a[k]:
                yRep.insert(k,a[k])
            elif y[k] > b[k]:
                yRep.insert(k,b[k])
            else:
                yRep.insert(k,y[k])

    yRep.extend(0 for j in range(actual_length_y - x_count))  # Append extra zeros
    return yRep

def Select_Random(Q, population, i, x):
    '''
    :param mapped_matrix: Codebook Matrix after one to one mapping
    :param population: archive Population
    :param Q: Mating Pool
    :param x: current solution
    :return: Random two parents from Q or population based on flag value
    '''


    my_randoms = random.sample(xrange(0, len(Q)), 3)
    p1 = population[my_randoms[0]]
    p2=population[my_randoms[1]]
    p3=population[my_randoms[2]]
    return p1,p2,p3








def Mutation(yRval,di,xmax,xmin):
    '''

    :param yRval: Value at ith index of Repaired solution
    :param x: Current population
    :param di: Mutated value
    :return:
    '''
    nsol=(yRval + di * (xmax - xmin))
    #print "nlsolution part : ", nsol

    return nsol

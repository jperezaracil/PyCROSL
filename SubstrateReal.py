import numpy as np
import random
from Substrate import *


"""
Substrate class that has continuous mutation and cross methods
"""
class SubstrateReal(Substrate):
    def __init__(self, evolution_method):
        self.evolution_method = evolution_method
        super().__init__(self.evolution_method)
    
    """
    Evolves a solution with a different strategy depending on the type of substrate
    """
    def evolve(self, solution, strength, population):
        result = None
        solution2 = random.choice([i for i in population if (i != solution).any()])
        if self.evolution_method == "1point":
            result = cross1p(solution, solution2)
        elif self.evolution_method == "2point":
            result = cross2p(solution, solution2)
        elif self.evolution_method == "Multipoint":
            result = crossMp(solution, solution2)
        elif self.evolution_method == "Gauss":
            result = gaussian(solution, strength)
        elif self.evolution_method == "Perm":
            result = permutation(solution, strength)
        else:
            print("Error: evolution method not defined")
            exit(1)
        
        return result

## Mutation and recombination methods
def gaussian(solution, strength):
    return solution + np.random.normal(0,strength,solution.shape)

def permutation(solution, strength):
    mask = np.random.random(solution.shape) < strength
    np.random.shuffle(solution[mask])
    return solution

def cross1p(solution1, solution2):
    cross_point = random.randrange(0, solution1.size)
    return np.hstack([solution1[:cross_point], solution2[cross_point:]])

def cross2p(solution1, solution2):
    cross_point1 = random.randrange(0, solution1.size-2)
    cross_point2 = random.randrange(cross_point1, solution1.size)
    return np.hstack([solution1[:cross_point1], solution2[cross_point1:cross_point2], solution1[cross_point2:]])

def crossMp(solution1, solution2):
    vector = 1*(np.random.rand(solution1.size) > 0.5)
    aux = np.copy(solution1)
    aux[vector==1] = solution2[vector==1]
    return aux

def bxalpha(self, solution1, solution2):
    p = np.random.rand()
    return (p*solution1 + (1-p)*solution2).astype(np.int32)

def sxb(self, solution):
    pass

def DERand1(self, solution):
    pass

def DEBest1(self, solution):
    pass

def DERand2(self, solution):
    pass

def DECurrentToBest1(self, solution):
    pass

def DECurrentToRand1(self, solution):
    pass
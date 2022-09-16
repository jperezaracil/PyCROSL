import numpy as np
import random
from Substrate import *

"""
Substrate class that has discrete mutation and cross methods
"""
class SubstrateInt(Substrate):
    def __init__(self, evolution_method):
        self.evolution_method = evolution_method
        super().__init__(self.evolution_method)
    
    """
    Applies a mutation method depending on the type of substrate
    """
    def evolve(self, solution, strength, population):
        result = None
        population.remove(solution)
        solution2 = random.choice(population)
        if self.evolution_method == "1point":
            result = cross1p(solution, solution2)
        elif self.evolution_method == "2point":
            result = cross2p(solution, solution2)
        elif self.evolution_method == "Multipoint":
            result = crossMp(solution, solution2)
        elif self.evolution_method == "BLXalpha":
            retult = bxalpha(solution1, solution2)
        elif self.evolution_method == "Perm":
            result = permutationMutation(solution, strength)
        elif self.evolution_method == "Xor":
            result = xorMaskMutation(solution, strength)
        elif self.evolution_method == "DGauss":
            result = discreteGaussianMutation(solution, strength)
        elif self.evolution_method == "AddOne":
            result = addOneMutation(solution, strength)
        else:
            print("Error: mutation method not defined")
            exit(1)

        return result

## Mutation and recombination methods
def discreteGaussian(solution, strength):
    return (solution + np.random.normal(0,100*strength,solution.shape)).astype(np.int32)

def addOne(solution, strength):
    increase = np.random.choice([-1,0,1], size=solution.size, p=[strength/2, 1-strength, strength/2])
    return (solution + increase).astype(np.int32)

def xorMask(solution, strength):
    if min(solution) == 0 and max(solution) == 1:
        vector = np.random.random(solution.shape) < strength
    else:
        mask = (np.random.random(solution.shape) < strength).astype(np.int32)
        vector = np.random.randint(1, 0xFF, size=solution.shape) * mask
    return (solution ^ vector).astype(np.int32)

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

def blxalpha(solution1, solution2):
    p = np.random.rand()
    return p*solution1 + (1-p)*solution2

def sxb(solution):
    pass

def DERand1(solution):
    pass

def DEBest1(solution):
    pass

def DERand2(solution):
    pass

def DECurrentToBest1(solution):
    pass

def DECurrentToRand1(solution):
    pass

# https://www.frontiersin.org/articles/10.3389/fbuil.2020.00102/full
# https://www.researchgate.net/publication/221008091_Modified_SBX_and_adaptive_mutation_for_real_world_single_objective_optimization
# https://stackoverflow.com/questions/38952879/blx-alpha-crossover-what-approach-is-the-right-one
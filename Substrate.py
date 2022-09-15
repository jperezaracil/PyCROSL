import numpy as np
import random

"""
Abstract Substrate class
"""
class Substrate:
    def __init__(self, evolution_method):
        self.evolution_method = evolution_method
    
    """
    Evolves a solution with a different strategy depending on the type of substrate
    """
    def evolve(self, solution, strength, population):
        pass

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
        if self.evolution_method == "1point":
            solution2 = random.choice(population)
            result = cross1p(solution, solution2)
        elif self.evolution_method == "2point":
            solution2 = random.choice(population)
            result = cross2p(solution, solution2)
        elif self.evolution_method == "Multipoint":
            solution2 = random.choice(population)
            result = crossMp(solution, solution2)
        elif self.evolution_method == "Gauss":
            result = gaussianMutation(solution, strength)
        elif self.evolution_method == "Perm":
            result = permutationMutation(solution, strength)
        else:
            print("Error: mutation method not defined")
            exit(1)
        
        return result

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
        if self.evolution_method == "1point":
            solution2 = random.choice(population)
            result = cross1p(solution, solution2)
        elif self.evolution_method == "2point":
            solution2 = random.choice(population)
            result = cross2p(solution, solution2)
        elif self.evolution_method == "Multipoint":
            solution2 = random.choice(population)
            result = crossMp(solution, solution2)
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
        

## Mutation methods
def gaussianMutation(solution, strength):
    return solution + np.random.normal(0,strength,solution.shape)

def discreteGaussianMutation(solution, strength):
    return (solution + np.random.normal(0,100*strength,solution.shape)).astype(np.int32)

def addOneMutation(solution, strength):
    increase = (np.random.rand(solution.size) > strength).astype(np.int32)
    return (solution + increase).astype(np.int32)

def xorMaskMutation(solution, strength):
    mask = np.random.random(solution.shape) < strength
    return (solution ^ mask).astype(np.int32)

def permutationMutation(solution, strength):
    mask = np.random.random(solution.shape) < strength
    np.random.shuffle(solution[mask])
    return solution

## Cross methods
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

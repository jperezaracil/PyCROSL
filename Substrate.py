import numpy as np
import random

"""
Abstract Substrate class
"""
class Substrate:
    def __init__(self, mutation_method, cross_method):
        self.mutation_method = mutation_method
        self.cross_method = cross_method
    
    """
    Applies a mutation method depending on the type of substrate
    """
    def mutate(self, solution, strength):
        pass
    
    """
    Applies a crossing method depending on the type of substrate.

    Methods common for real and discrete encoding.
    """
    def cross(self, solution1, solution2):
        result = None
        if self.cross_method == "1point":
            result = cross1p(solution1, solution2)
        elif self.cross_method == "2point":
            result = cross2p(solution1, solution2)
        elif self.cross_method == "Multipoint":
            result = crossMp(solution1, solution2)
        else:
            print("Error: cross method not defined")
            exit(1)
        
        return result

"""
Substrate class that has continuous mutation and cross methods
"""
class SubstrateReal(Substrate):
    def __init__(self, mutation_method, cross_method):
        self.mutation_method = mutation_method
        self.cross_method = cross_method
        super().__init__(self.mutation_method, self.cross_method)
    
    """
    Applies a mutation method depending on the type of substrate
    """
    def mutate(self, solution, strength):
        result = None
        if self.mutation_method == "Gauss":
            result = gaussianMutation(solution, strength)
        elif self.mutation_method == "Perm":
            result = permutationMutation(solution, strength)
        else:
            print("Error: mutation method not defined")
            exit(1)
        
        return result

"""
Substrate class that has discrete mutation and cross methods
"""
class SubstrateInt(Substrate):
    def __init__(self, mutation_method, cross_method):
        self.mutation_method = mutation_method
        self.cross_method = cross_method
        super().__init__(self.mutation_method, self.cross_method)
    
    """
    Applies a mutation method depending on the type of substrate
    """
    def mutate(self, solution, strength):
        result = None
        if self.mutation_method == "Perm":
            result = permutationMutation(solution, strength)
        elif self.mutation_method == "Xor":
            result = xorMaskMutation(solution, strength)
        else:
            print("Error: mutation method not defined")
            exit(1)

        return result
        

## Mutation methods
def gaussianMutation(solution, strength):
    return solution + np.random.normal(0,strength,solution.shape)

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

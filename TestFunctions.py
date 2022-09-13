import random
import numpy as np
from AbsObjetiveFunc import AbsObjetiveFunc

"""
Example of objective function.

Counts the number of ones in the array
"""
class MaxOnes(AbsObjetiveFunc):
    def __init__(self, size):
        self.size = size
        super().__init__(self.size)

    def fitness(self, solution):
        return solution.sum()
    
    def random_solution(self):
        return (np.random.random(self.size) < 0.5).astype(np.int32)
    
    def check_bounds(self, solution):
        return np.equal(solution.copy(),0).astype(np.int32)

class DiophantineEq(AbsObjetiveFunc):
    def __init__(self, size, coeff, target):
        self.size = size
        self.coeff = coeff
        self.target = target
        super().__init__(self.size)

    def fitness(self, solution):
        return abs((solution*self.coeff).sum() - self.target)
    
    def random_solution(self):
        return (np.random.randint(-100, 100, size=self.size)).astype(np.int32)
    
    def check_bounds(self, solution):
        return solution.astype(np.int32)

class MaxOnesReal(AbsObjetiveFunc):
    def __init__(self, size):
        self.size = size
        super().__init__(self.size)

    def fitness(self, solution):
        return solution.sum()
    
    def random_solution(self):
        return np.random.random(self.size)
    
    def check_bounds(self, solution):
        return np.clip(solution.copy(),0,1)

# https://www.scientificbulletin.upb.ro/rev_docs_arhiva/rez0cb_759909.pdf

class Sphere(AbsObjetiveFunc):
    def __init__(self, size):
        self.size = size
        super().__init__(self.size)

    def fitness(self, solution):
        return (solution**2).sum()
    
    def random_solution(self):
        return np.random.random(self.size)*100
    
    def check_bounds(self, solution):
        return np.clip(solution, -100, 100)

class Test1(AbsObjetiveFunc):
    def __init__(self, size):
        self.size = size
        super().__init__(self.size)

    def fitness(self, solution):
        return sum([(2*solution[i-1] + solution[i]**2*solution[i+1]-solution[i-1])**2 for i in range(1, solution.size-1)])
    
    def random_solution(self):
        return np.random.random(self.size)*2
    
    def check_bounds(self, solution):
        return np.clip(solution, -2, 2)

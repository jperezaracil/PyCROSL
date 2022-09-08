import random
import numpy as np
from AbsObjetiveFunc import AbsObjetiveFunc

"""
Example of objective function.

Counts the number of ones in the array
"""
class ExampleObj(AbsObjetiveFunc):
    def __init__(self, size):
        self.size = size
        super().__init__(self.size)

    def fitness(self, solution):
        return solution.sum()
    
    def random_solution(self):
        one_ratio = random.random()
        return (np.random.random(self.size) < one_ratio).astype(np.int32)
    
    def mutate(self, solution, strength):
        mask = (np.random.random(self.size) < strength).astype(np.int32)
        return solution ^ mask
    
    def cross2p(self, solution1, solution2):
        return 0,0
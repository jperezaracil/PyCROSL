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
        return (np.random.random(self.size) < 0.5).astype(np.int32)
    
    def check_bounds(self, solution):
        return np.equal(solution.copy(),0).astype(np.int32)
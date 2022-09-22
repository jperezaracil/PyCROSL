"""
Abstract Fitness function class.

For each problem a new class will inherit from this one
and implement the fitness function, random solution generation,
mutation function and crossing of solutions.
"""
class AbsObjetiveFunc:
    def __init__(self, input_size, opt):
        self.counter = 0
        self.input_size = input_size
        self.factor = 1
        self.opt = opt
        if self.opt == "min":
            self.factor = -1
    
    def fitness(self, solution):
        self.counter += 1
    
    def random_solution(self):
        pass
    
    def check_bounds(self, solution):
        pass

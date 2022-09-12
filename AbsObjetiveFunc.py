"""
Abstract Fitness function class.

For each problem a new class will inherit from this one
and implement the fitness function, random solution generation,
mutation function and crossing of solutions.
"""
class AbsObjetiveFunc:
    def __init__(self, input_size):
        self.input_size = input_size
    
    def fitness(self, solution):
        pass
    
    def random_solution(self):
        pass
    
    def check_bounds(self, solution):
        pass

"""
Abstract Fitness function class.

For each problem a new class will inherit from this one
and implement the fitness function, random solution generation,
mutation function and crossing of solutions.
"""
class AbsObjectiveFunc:
    def __init__(self, input_size, opt, sup_lim = 100, inf_lim = -100):
        self.counter = 0
        self.input_size = input_size
        self.factor = 1
        self.opt = opt
        self.sup_lim = sup_lim
        self.inf_lim = inf_lim
        if self.opt == "min":
            self.factor = -1
    
    def fitness(self, solution):
        self.counter += 1
        return self.factor * self.objective(solution)
    
    def objective(self, solution):
        pass
    
    def random_solution(self):
        pass
    
    def check_bounds(self, solution):
        pass
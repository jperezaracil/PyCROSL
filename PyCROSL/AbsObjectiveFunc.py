from abc import ABC, abstractmethod

class AbsObjectiveFunc(ABC):
    """
    Abstract Fitness function class.

    For each problem a new class will inherit from this one
    and implement the fitness function, random solution generation,
    mutation function and crossing of solutions.
    """

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
        """
        Returns an adjusted version of the objective functio so that the problem becomes
        a maximization problem.
        """

        self.counter += 1
        return self.factor * self.objective(solution)
    
    @abstractmethod
    def objective(self, solution):
        """
        Implementation of the objective function.
        """
    
    @abstractmethod
    def random_solution(self):
        """
        Returns a random vector. 
        """
    
    @abstractmethod
    def repair_solution(self, solution):
        """
        Modifies a solution so that it satisfies the restrictions of the problem.
        """
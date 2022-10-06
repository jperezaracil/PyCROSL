import random
import numpy as np
from numba import jit

"""
Individual that holds a tentative solution with 
its fitness.
"""
class Indiv:
    """
    Constructor
    """
    def __init__(self, solution, objfunc):
        self.solution = solution
        
        self.fitness_calculated = False
        self.fitness = 0

        self.objfunc = objfunc

    def get_fitness(self):
        if not self.fitness_calculated:
            self.fitness = self.objfunc.fitness(self.solution)
            self.fitness_calculated = True
        return self.fitness


"""
Population of individuals
"""
class Population:    
    """
    Constructor of the Population class
    """
    def __init__(self, objfunc, mutation_op, replace_op, params, population=None):
        # Hyperparameters of the algorithm
        self.size = params["PopSize"]
        self.hmcr = params["HMCR"]
        self.par = params["PAR"]
        self.bn = params["BN"]
        self.mutation_op = mutation_op
        self.replace_op = replace_op
        self.mutation_levels = len(mutation_op)

        # Data structures of the algorithm
        self.objfunc = objfunc

        # Population initialization
        if population is None:
            self.population = []
        else:
            self.population = population
        self.offspring = []       

    """
    Gives the best solution found by the algorithm and its fitness
    """
    def best_solution(self):
        best_solution = sorted(self.population, reverse=True, key = lambda c: c.get_fitness())[0]
        best_fitness = best_solution.get_fitness()
        if self.objfunc.opt == "min":
            best_fitness *= -1
        return (best_solution.solution, best_fitness)

    """
    Generates a random population of individuals
    """
    def generate_random(self):
        self.population = []
        for i in range(self.size):
            new_ind = Indiv(self.objfunc.random_solution(), self.objfunc)
            self.population.append(new_ind)
    
    """
    Applies the HS operation to the population
    """
    def evolve(self):
        # popul_matrix = HM
        popul_matrix = np.vstack([i.solution for i in self.population])
        solution_size = popul_matrix.shape[1]
        new_solution = np.zeros(solution_size)
        mask1 = np.zeros(solution_size)
        mask2 = np.ones(solution_size)
        mask3 = -np.ones(solution_size)
        popul_mean = popul_matrix.mean(axis=0)
        for i in range(solution_size):
            if random.random() < self.hmcr:
                pos = random.randrange(popul_matrix.shape[0])
                new_solution[i] = popul_matrix[pos][i]
                mask2[i] = 0
                mask3[i] = pos
                if random.random() <= self.par:
                    mask1[i] = 1
        mask1 = mask1 >=1
        mask2 = mask2 >=1
        for i in range(self.mutation_levels):
            step = solution_size/self.mutation_levels
            mask1_aux = (mask3 >= int(step*i)) & (mask3 < int(step*i + step)) & mask1
            new_solution[mask1_aux] = self.mutation_op[i].evolve(Indiv(new_solution, self.objfunc), self.population, self.objfunc)[mask1_aux]
            new_solution[mask2] = self.replace_op.evolve(Indiv(new_solution, self.objfunc), self.population, self.objfunc)[mask2]
        
        new_solution = self.objfunc.check_bounds(new_solution)
        self.population.append(Indiv(new_solution, self.objfunc))
    
    """
    Removes the worse solutions of the population
    """
    def selection(self):
        actual_size_pop = len(self.population)
        fitness_values = np.array([ind.get_fitness() for ind in self.population])
        #print(fitness_values)
        kept_ind = list(np.argsort(fitness_values))[:(actual_size_pop - self.size)]

        self.population = [self.population[i] for i in range(len(self.population)) if i not in kept_ind] 


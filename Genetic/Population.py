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
    def __init__(self, solution, objfunc, operator=None):
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
    def __init__(self, objfunc, mutation_op, cross_op, params, population=None):
        # Hyperparameters of the algorithm
        self.size = params["PopSize"]
        self.pmut = params["Pmut"]
        self.mutation_op = mutation_op
        self.cross_op = cross_op

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
    Applies a random mutation to a small portion of individuals
    """
    def mutate(self):
        for i in self.offspring:
            new_ind = i
            if random.random() < self.pmut:
                new_solution = self.mutation_op.evolve(new_ind, self.population, self.objfunc)
                new_ind = Indiv(new_solution, self.objfunc)
            self.population.append(new_ind)
    
    """
    Crosses individuals of the population
    """
    def cross(self):
        self.offspring = []
        for i in range(len(self.population)):
            parent1 = random.choice(self.population)
            new_solution = self.cross_op.evolve(parent1, self.population, self.objfunc)
            self.offspring.append(Indiv(new_solution, self.objfunc))
    
    """
    Removes the worse solutions of the population
    """
    def selection(self):
        actual_size_pop = len(self.population)
        fitness_values = np.array([ind.get_fitness() for ind in self.population])
        affected_ind = list(np.argsort(fitness_values))[:(actual_size_pop - self.size)]

        self.population = [self.population[i] for i in range(len(self.population)) if i not in affected_ind] 

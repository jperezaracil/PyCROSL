import random
import numpy as np
from Substrate import *

"""
Individual that holds a tentative solution with 
its fitness.
"""
class Coral:
    """
    Constructor
    """
    def __init__(self, solution, objfunc, opt, substrate=None):
        self.solution = solution
        
        self.fitness_calculated = False
        self.fitness = 0

        self.is_dead = False

        self.objfunc = objfunc

        self.opt = opt

        self.substrate = substrate

    def get_fitness(self):
        if not self.fitness_calculated:
            self.fitness = self.objfunc.fitness(self.solution)
            if self.opt == "min":
                self.fitness *= -1
            self.fitness_calculated = True
        return self.fitness
    
    def set_substrate(self, substrate):
        self.substrate = substrate
    
    def reproduce(self, strength, population):
        new_solution = self.substrate.evolve(self.solution.copy(), strength, [c.solution.copy() for c in population])
        new_solution = self.objfunc.check_bounds(new_solution)
        return Coral(new_solution, self.objfunc, self.opt)


"""
Population of corals
"""
class CoralPopulation:    
    def __init__(self, size, objfunc, substrates, opt, population=None):
        self.size = size
        self.objfunc = objfunc
        self.substrates = substrates
        self.fitness_count = 0
        self.opt = opt

        if population is None:
            self.population = []
        
        self.substrate_list = [i%len(substrates) for i in range(self.size)]

    def best_solution(self):
        best_solution = sorted(self.population, reverse=True, key = lambda c: c.get_fitness())[0]
        return (best_solution.solution, best_solution.get_fitness())

    def generate_random(self, proportion):
        amount = int(self.size*proportion)
        self.population = []
        for i in range(amount):
            substrate_idx = self.substrate_list[i]
            new_coral = Coral(self.objfunc.random_solution(), self.objfunc, self.opt, self.substrates[substrate_idx])
            new_coral.get_fitness()
            self.fitness_count += 1
            self.population.append(new_coral)
    
    def evolve_with_substrates(self, mut_strength):
        # Divide the population based on their substrate type
        substrate_groups = [[] for i in self.substrates]
        for i, idx in enumerate(self.substrate_list):
            if i < len(self.population):
                substrate_groups[idx].append(self.population[i])
        
        # Reproduce the corals of each group
        larvae = []        
        for coral_group in substrate_groups:
            for coral in coral_group:
                # Generate new coral
                new_coral = coral.reproduce(mut_strength, coral_group.copy())
        
                # Evaluate it's fitness
                new_coral.get_fitness()
                self.fitness_count += 1
        
                # Add larva to the list of larvae
                larvae.append(new_coral)
        
        return larvae

    
    def generate_substrates(self):
        self.substrate_list = [random.randrange(0,len(self.substrates)) for i in range(self.size)]

    def larvae_setting(self, larvae_list, attempts):
        for larva in larvae_list:
            attempts_left = attempts
            setted = False
            idx = -1

            # Try to settle 'attempts' times
            while attempts_left > 0 and not setted:
                # Choose a random position
                idx = random.randrange(0, self.size)

                # If it's empty settle in there if it's not the case,
                # try to replace the coral in that position
                if idx >= len(self.population):
                    setted = True
                    self.population.append(larva)
                elif setted := (larva.fitness > self.population[idx].fitness):
                    self.population[idx] = larva

                attempts_left -= 1
            
            # Assign substrate to the setted coral
            if setted:
                substrate_idx = self.substrate_list[idx]
                larva.set_substrate(self.substrates[substrate_idx])
        
    
    def budding(self, proportion, attempts):
        n_corals = int(self.size*proportion)

        corals = self.population.copy()
        duplicates = sorted(self.population, reverse=True, key = lambda c: c.get_fitness())[:n_corals]

        self.larvae_setting(duplicates, attempts)
    
    def depredation(self, proportion, probability):
        # Calculate the number of affected corals
        amount = int(self.size*proportion)

        # Extract the worse 'n_corals' in the grid
        fitness_values = np.array([coral.get_fitness() for coral in self.population])
        affected_corals = list(np.argsort(fitness_values))[:amount]

        # Set a 'dead' flag in the affected corals with a small probability
        for i in affected_corals:
            self.population[i].is_dead = random.random() <= probability

        # Remove the dead corals from the population
        self.population = list(filter(lambda c: not c.is_dead, self.population))

    def extreme_depredation(self):
        pass    


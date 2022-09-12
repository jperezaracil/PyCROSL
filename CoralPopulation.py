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
    def __init__(self, solution, objfunc, substrate=None):
        self.solution = solution
        
        self.fitness_calculated = False
        self.fitness = 0

        self.is_dead = False

        self.objfunc = objfunc

        self.substrate = substrate

    def get_fitness(self):
        if not self.fitness_calculated:
            self.fitness = self.objfunc.fitness(self.solution)
            self.fitness_calculated = True
        return self.fitness
    
    def set_substrate(self, substrate):
        self.substrate = substrate
    
    def mutate(self, strength):
        mutated_solution = self.substrate.mutate(self.solution.copy(), strength)
        mutated_solution = self.objfunc.check_bounds(mutated_solution)
        return Coral(mutated_solution, self.objfunc)
    
    def cross(self, indiv):
        crossed_solution = self.substrate.cross(self.solution.copy(), indiv.solution.copy())
        crossed_solution = self.objfunc.check_bounds(crossed_solution)
        return Coral(crossed_solution, self.objfunc)

"""
Population of corals
"""
class CoralPopulation:    
    def __init__(self, size, objfunc, substrates, population=None):
        self.size = size
        self.objfunc = objfunc
        self.substrates = substrates

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
            self.population.append(Coral(self.objfunc.random_solution(), self.objfunc, self.substrates[substrate_idx]))

    def broadcast_spawning(self, proportion):
        # Calculate the number of affected corals
        n_corals = int(len(self.population)*proportion)

        # Force n_corals to an even number
        n_corals = (n_corals//2)*2

        # Choose indices of the corals to be broadcast
        corals_chosen = list(range(len(self.population)))
        random.shuffle(corals_chosen)
        corals_chosen = corals_chosen[:n_corals]
        corals_chosen_aux = corals_chosen.copy()

        # Generate larvae
        larvae = []
        while len(corals_chosen) > 1:
            # Take 2 random parents and remove them from the pool
            parent1_idx = corals_chosen.pop(random.randrange(0, len(corals_chosen)))
            parent2_idx = corals_chosen.pop(random.randrange(0, len(corals_chosen)))

            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]

            # Generate a new larva crossing the parents
            larvae.append(parent1.cross(parent2))

        return larvae, corals_chosen_aux
    
    def brooding(self, corals_chosen_prev, mut_strength):
        # Choose the corals that weren't chosen in the previous step
        corals_chosen = [i for i in range(len(self.population)) if i not in corals_chosen_prev]

        # Mutate the corals chosen and create larvae
        larvae = []
        for i in corals_chosen:
            larvae.append(self.population[i].mutate(mut_strength))
        
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
        duplicates = sorted(self.population, reverse=True, key = lambda c: c.get_fitness())[:min(n_corals, len(self.population))]

        self.larvae_setting(duplicates, attempts)
    
    def depredation(self, proportion, probability):
        # Calculate the number of affected corals
        amount = int(self.size*proportion)

        # Extract the worse 'n_corals' in the grid
        fitness_values = np.array([coral.get_fitness() for coral in self.population])
        affected_corals = list(np.argsort(fitness_values))[:min(amount, len(self.population))]

        # Set a 'dead' flag in the affected corals with a small probability
        for i in affected_corals:
            self.population[i].is_dead = random.random() <= probability

        # Remove the dead corals from the population
        self.population = list(filter(lambda c: not c.is_dead, self.population))

    def extreme_depredation(self):
        pass    


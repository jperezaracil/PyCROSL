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
            self.fitness_calculated = True
        return self.fitness
    
    def set_substrate(self, substrate):
        self.substrate = substrate
    
    def reproduce(self, strength, population):
        new_solution = self.substrate.evolve(self, strength, population, self.objfunc)
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
        self.fit_improvement = [0]*len(substrates)
        self.substrate_history = []
        self.substrate_weight = [1]*len(substrates)

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
            self.population.append(new_coral)
    
    def evolve_with_substrates(self, mut_strength):
        # Divide the population based on their substrate type
        substrate_groups = [[] for i in self.substrates]
        for i, idx in enumerate(self.substrate_list):
            if i < len(self.population):
                substrate_groups[idx].append(self.population[i])
        
        
        # Reproduce the corals of each group
        larvae = []        
        for i, coral_group in enumerate(substrate_groups):
            for coral in coral_group:
                # Generate new coral
                new_coral = coral.reproduce(mut_strength, coral_group)
                
                # Update the fitness improvement record
                self.fit_improvement[i] += new_coral.get_fitness()/len(coral_group)

                # Add larva to the list of larvae
                larvae.append(new_coral)
        return larvae

    def amplify_probability(self, values):
        weight = np.array(self.fit_improvement)
        weight = weight/weight.sum()
        weight = np.exp(weight)

        if self.opt == "min":
            weight = 1/weight

        weight -= 9*min(weight)/10
        for i in range(7):
            weight = np.log(1-weight)/np.log(1-weight).sum()
        return weight

    def generate_substrates(self):
        n_substrates = len(self.substrates)

        # Assign the probability of each substrate
        weight = np.ones(n_substrates)/n_substrates
        if sum(self.fit_improvement) != 0:
            # Normalize the fitness record of each substrate
            weight = self.amplify_probability(weight)
           
            print(weight)

        self.substrate_weight = weight
        
        # Choose each substrate with the weights chosen
        self.substrate_list = random.choices(range(n_substrates), weights=weight, k=self.size)

        # Assign the substrate to each coral
        for idx, coral in enumerate(self.population):
            substrate_idx = self.substrate_list[idx]
            coral.set_substrate(self.substrates[substrate_idx])

        self.substrate_history.append(np.array(self.fit_improvement))
        # Reset fitness improvement record
        self.fit_improvement = [0]*n_substrates

    def larvae_setting(self, larvae_list, attempts):
        for larva in larvae_list:
            attempts_left = attempts
            setted = False
            idx = -1

            # Try to settle 'attempts times
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
        if len(self.population) <= self.size*0.2:
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
        alive_count = len(self.population)
        for i in affected_corals:
            if alive_count <= 2:
                break
            dies = random.random() <= probability
            self.population[i].is_dead = dies
            if dies:
                alive_count -= 1

        # Remove the dead corals from the population
        self.population = list(filter(lambda c: not c.is_dead, self.population))

    def extreme_depredation(self):
        pass    


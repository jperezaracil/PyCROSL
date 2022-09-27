import random
import numpy as np
import warnings
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
    
    def reproduce(self, population):
        new_solution = self.substrate.evolve(self, population, self.objfunc)
        new_solution = self.objfunc.check_bounds(new_solution)
        return Coral(new_solution, self.objfunc, self.opt)


"""
Population of corals
"""
class CoralPopulation:    
    def __init__(self, size, objfunc, substrates, dyn_metric, prob_amp, population=None):
        self.size = size
        self.objfunc = objfunc
        self.substrates = substrates
        self.dyn_metric = dyn_metric
        self.prob_amp = prob_amp
        self.substrate_metric = [0]*len(substrates)
        self.substrate_history = []
        self.substrate_weight = [1]*len(substrates)
        self.substrate_w_history = [[1/len(substrates)]*len(substrates)]

        self.prob_amp_warned = False

        self.opt = objfunc.opt

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
    
    def evaluate_substrate(self, idx, fit_list):
        if len(fit_list) > 0:
            if self.dyn_metric == "max":
                self.substrate_metric[idx] = max(fit_list)
            elif self.dyn_metric == "avg":
                self.substrate_metric[idx] = sum(fit_list)/len(fit_list)
            elif self.dyn_metric == "med":
                if len(fit_list) % 2 == 0:
                    self.substrate_metric[idx] = fit_list[len(fit_list)//2-1]+fit_list[len(fit_list)//2]
                else:
                    self.substrate_metric[idx] = fit_list[len(fit_list)//2]
            elif self.dyn_metric == "min":
                self.substrate_metric[idx] = min(fit_list)
        else:
            self.substrate_metric[idx] = 0



    def evolve_with_substrates(self):
        # Divide the population based on their substrate type
        substrate_groups = [[] for i in self.substrates]
        for i, idx in enumerate(self.substrate_list):
            if i < len(self.population):
                substrate_groups[idx].append(self.population[i])
        
        
        # Reproduce the corals of each group
        larvae = []        
        for i, coral_group in enumerate(substrate_groups):
            # Restart fitness record if there are corals in this substrate
            #if len(coral_group) != 0:
            #    self.substrate_metric[i] = 0

            substrate_fitness_list = []

            for coral in coral_group:
                # Generate new coral
                new_coral = coral.reproduce(coral_group)
                
                # Update the fitness improvement record
                #self.substrate_metric[i] += new_coral.get_fitness()/len(coral_group)
                substrate_fitness_list.append(new_coral.get_fitness())

                # Add larva to the list of larvae
                larvae.append(new_coral)
            
            self.evaluate_substrate(i, substrate_fitness_list)
            
        return larvae

    def amplify_probability(self, values):
        # Normalization
        weight = np.array(values)
        if weight.sum() != 0:
            weight = weight/np.abs(weight.sum())
        #print(weight)
        
        # softmax to convert to a probability distribution
        exp_vec = np.exp(weight)
        amplified_vec = exp_vec**(1/self.prob_amp)
        

        if (amplified_vec == 0).any():
            if not self.prob_amp_warned:
                print("Warning: the probability amplification parameter is too small, defaulting to prob_amp = 1")
                self.prob_amp_warned = True
            prob = exp_vec/exp_vec.sum()
        else:
            prob = amplified_vec/amplified_vec.sum()
            
        
        if (prob <= 0.00001).any():
            prob += 0.02/len(values)
            prob = prob/prob.sum()

        return prob

    def generate_substrates(self):
        n_substrates = len(self.substrates)

        # Assign the probability of each substrate
        self.substrate_weight = self.amplify_probability(self.substrate_metric)
        self.substrate_w_history.append(self.substrate_weight)
        
        # Choose each substrate with the weights chosen
        self.substrate_list = random.choices(range(n_substrates), 
                                             weights=self.substrate_weight, k=self.size)

        # Assign the substrate to each coral
        for idx, coral in enumerate(self.population):
            substrate_idx = self.substrate_list[idx]
            coral.set_substrate(self.substrates[substrate_idx])

        self.substrate_history.append(np.array(self.substrate_metric))

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

    def extreme_depredation(self, tol=0):
        solutions = [i.solution for i in self.population]

        repeated_idx = []
        for idx, val in enumerate(solutions):
            if any([(np.abs(i-val) < tol).all() for i in solutions[:idx]]):
                repeated_idx.append(idx)

        self.population = [val for idx, val in enumerate(self.population) if idx not in repeated_idx]


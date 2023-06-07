import random

import numpy as np
from numba import jit
from PyCROSL.operators import *
from joblib import Parallel, delayed 


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
    
    def reproduce(self, population):
        new_solution = self.substrate.evolve(self, population, self.objfunc)
        new_solution = self.objfunc.repair_solution(new_solution)
        return Coral(new_solution, self.objfunc, self.substrate)


"""
Population of corals
"""
class CoralPopulation:    
    """
    Constructor of the Coral Population class
    """
    def __init__(self, objfunc, substrates, params, population=None):
        # Hyperparameters of the algorithm
        self.size = params["popSize"]
        self.rho = params["rho"]
        self.Fb = params["Fb"]
        self.Fd = params["Fd"]
        self.Pd = params["Pd"]
        self.k = params["k"]
        self.K = params["K"]
        self.group_subs = params["group_subs"]

        # Dynamic parameters
        self.dynamic = params["dynamic"]
        self.dyn_method = params["dyn_method"]
        self.dyn_metric = params["dyn_metric"]
        self.dyn_steps = params["dyn_steps"]
        self.prob_amp = params["prob_amp"]

        # Verbose parameters
        self.verbose = params["verbose"]
        self.v_timer = params["v_timer"]

        # Data structures of the algorithm
        self.objfunc = objfunc
        self.substrates = substrates

        # Population initialization
        if population is None:
            self.population = []

        # Substrate data structures
        self.substrate_list = [i%len(substrates) for i in range(self.size)]
        self.substrate_weight = [1/len(substrates)]*len(substrates)

        # Dynamic data structures
        self.substrate_data = [[] for i in substrates]
        if self.dyn_method == "success":
            for idx, _ in enumerate(self.substrate_data):
                self.substrate_data[idx].append(0)
            self.larva_count = [0 for i in substrates]
        elif self.dyn_method == "diff":
            self.substrate_metric_prev = [0]*len(substrates)
        self.substrate_w_history = []
        self.subs_steps = 0
        self.substrate_metric = [0]*len(substrates)
        self.substrate_history = []

        self.prob_amp_warned = False

        # Optimization for extreme depredation
        self.updated = False
        self.identifier_list = []

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
    Generates a random population of corals
    """
    def generate_random(self):
        amount = int(self.size*self.rho) - len(self.population)

        for i in range(amount):
            substrate_idx = self.substrate_list[i]
            new_sol = self.objfunc.random_solution()
            fixed_sol = self.objfunc.repair_solution(new_sol)
            new_coral = Coral(fixed_sol, self.objfunc, self.substrates[substrate_idx])
            self.population.append(new_coral)

    
    def insert_solution(self, solution, mutate=False, strength=0.1):
        #solution = self.objfunc.repair_solution(solution)
        if mutate:
            solution = gaussian(solution, strength)
            solution = self.objfunc.repair_solution(solution)
        
        if len(self.population) < self.size:
            new_ind = Coral(solution, self.objfunc)
            self.population.append(new_ind)
        else:
            new_ind = Coral(solution, self.objfunc)
            idx = random.randint(self.size)
            self.population[idx] = new_ind
    
    def get_value_from_data(self, data):
        result = 0

        # Choose what information to extract from the data gathered
        if len(data) > 0:
            data = sorted(data)
            if self.dyn_metric == "best":
                result = max(data)
            elif self.dyn_metric == "avg":
                result = sum(data)/len(data)
            elif self.dyn_metric == "med":
                if len(data) % 2 == 0:
                    result = (data[len(data)//2-1]+data[len(data)//2])/2
                else:
                    result = data[len(data)//2]
            elif self.dyn_metric == "worse":
                result = min(data)
        
        return result

    """
    Evaluates the substrates using a given metric
    """
    def evaluate_substrates(self):
        metric = 0
        
        # take reference data for the calculation of the difference of the next evaluation
        if self.dyn_method == "diff":
            full_data = [d for subs_data in self.substrate_data for d in subs_data]
            metric = self.get_value_from_data(full_data)
        
        # calculate the value of each substrate with the data gathered
        for idx, s_data in enumerate(self.substrate_data):
            if self.dyn_method == "success":

                # obtain the rate of success of the larvae
                if self.larva_count[idx] > 0:
                    self.substrate_metric[idx] = s_data[0]/self.larva_count[idx]
                else:
                    self.substrate_metric[idx] = 0

                # Reset data for nex iteration
                self.substrate_data[idx] = [0]
                self.larva_count[idx] = 0

            elif self.dyn_method == "fitness" or self.dyn_method == "diff":

                # obtain the value used in the evaluation of the substrate 
                self.substrate_metric[idx] = self.get_value_from_data(s_data)

                # Calculate the difference of the fitness in this generation to the previous one and
                # store the current value for the next evaluation
                if self.dyn_method == "diff":
                    self.substrate_metric[idx] =  self.substrate_metric[idx] - self.substrate_metric_prev[idx]
                    self.substrate_metric_prev[idx] = metric
                
                # Reset data for next iteration
                self.substrate_data[idx] = []
                
        
        

    """
    Evolves the population using the corresponding substrates
    """
    def evolve_with_substrates(self):
        larvae = []

        # evolve within the substrate or mix with the whole population
        if self.group_subs:

            # Divide the population based on their substrate type
            substrate_groups = [[] for i in self.substrates]
            for i, idx in enumerate(self.substrate_list):
                if i < len(self.population):
                    substrate_groups[idx].append(self.population[i])
            
            
            # Reproduce the corals of each group
            for i, coral_group in enumerate(substrate_groups):

                # Restart fitness record if there are corals in this substrate
                for coral in coral_group:

                    # Generate new coral
                    if random.random() <= self.Fb:
                        new_coral = coral.reproduce(coral_group)
                        
                        # Get data of the current substrate
                        if self.dyn_method == "fitness" or self.dyn_method == "diff":
                            self.substrate_data[i].append(new_coral.get_fitness())
                    else:
                        new_sol = self.objfunc.random_solution()
                        fixed_sol = self.objfunc.repair_solution(new_sol)
                        new_coral = Coral(fixed_sol, self.objfunc)

                    # Add larva to the list of larvae
                    larvae.append(new_coral)
        else:
            for idx, coral in enumerate(self.population):

                # Generate new coral
                if random.random() <= self.Fb:
                    new_coral = coral.reproduce(self.population)
                    
                    # Get the index of the substrate this individual belongs to
                    s_names = [i.evolution_method for i in self.substrates]
                    s_idx = s_names.index(coral.substrate.evolution_method)

                    # Get data of the current substrate
                    if self.dyn_method == "fitness" or self.dyn_method == "diff":
                        self.substrate_data[s_idx].append(new_coral.get_fitness())
                else:
                    new_sol = self.objfunc.random_solution()
                    fixed_sol = self.objfunc.repair_solution(new_sol)
                    new_coral = Coral(fixed_sol, self.objfunc)

                # Add larva to the list of larvae
                larvae.append(new_coral)
        
        return larvae

    """
    Converts the evaluation values of the substrates to a probability distribution
    """
    def substrate_probability(self, values):

        # Normalization to avoid passing big values to softmax 
        weight = np.array(values)
        if np.abs(weight).sum() != 0:
            weight = weight/np.abs(weight).sum()
        else:
            weight = weight/(np.abs(weight).sum()+1e-5)
        
        # softmax to convert to a probability distribution
        exp_vec = np.exp(weight)
        amplified_vec = exp_vec**(1/self.prob_amp)
        
        # if there are numerical error default repeat with a default value
        if (amplified_vec == 0).any() or not np.isfinite(amplified_vec).all():
            if not self.prob_amp_warned:
                print("Warning: the probability amplification parameter is too small, defaulting to prob_amp = 1")
                self.prob_amp_warned = True
            prob = exp_vec/exp_vec.sum()
        else:
            prob = amplified_vec/amplified_vec.sum()

        # If probabilities get too low, equalize them
        if (prob <= 0.02/len(values)).any():
            prob += 0.02/len(values)
            prob = prob/prob.sum()

        return prob


    """
    Generates the assignment of the substrates
    """
    def generate_substrates(self, progress=0):
        n_substrates = len(self.substrates)

        if progress > self.subs_steps/self.dyn_steps:
            self.subs_steps += 1
            self.evaluate_substrates()

        # Assign the probability of each substrate
        if self.dynamic:
            self.substrate_weight = self.substrate_probability(self.substrate_metric)
            self.substrate_w_history.append(self.substrate_weight)
        
        # Choose each substrate with the weights chosen
        self.substrate_list = random.choices(range(n_substrates), 
                                            weights=self.substrate_weight, k=self.size)

        # Assign the substrate to each coral
        for idx, coral in enumerate(self.population):
            substrate_idx = self.substrate_list[idx]
            coral.set_substrate(self.substrates[substrate_idx])

        # save the evaluation of each substrate
        self.substrate_history.append(np.array(self.substrate_metric))


    """
    Inserts solutions into our reef with some conditions
    """
    def larvae_setting(self, larvae_list):
        s_names = [i.evolution_method for i in self.substrates]

        for larva in larvae_list:
            attempts_left = self.k
            setted = False
            idx = -1

            # Try to settle 
            while attempts_left > 0 and not setted:
                # Choose a random position
                idx = random.randrange(0, self.size)

                # If it's empty settle in there, otherwise, try
                # to replace the coral in that position
                if setted := (idx >= len(self.population)):
                    self.population.append(larva)
                elif setted := (larva.get_fitness() > self.population[idx].get_fitness()):
                    self.population[idx] = larva

                attempts_left -= 1
            
            if larva.substrate is not None:
                s_idx = s_names.index(larva.substrate.evolution_method)
                if self.dyn_method == "success":
                    self.larva_count[s_idx] += 1

            # Assign substrate to the setted coral
            if setted:
                self.updated = True
                
                # Get substrate index
                if self.dyn_method == "success" and larva.substrate is not None:
                    self.substrate_data[s_idx][0] += 1

                substrate_idx = self.substrate_list[idx]
                larva.set_substrate(self.substrates[substrate_idx])
    
    """
    Performs a local search with the best "n_ind" corals
    """
    def local_search(self, operator, n_ind, iterations=100):
        fitness_values = np.array([coral.get_fitness() for coral in self.population])
        affected_corals = list(np.argsort(fitness_values))[len(self.population)-n_ind:]
        
        for i in affected_corals:
            best = self.population[i]

            for j in range(iterations):
                new_solution = operator.evolve(self.population[i], [], self.objfunc)
                new_solution = self.objfunc.repair_solution(new_solution)
                new_coral = Coral(new_solution, self.objfunc, self.population[i].substrate)
                if new_coral.get_fitness() > best.get_fitness():
                    best = new_coral
            
            self.population[i] = best

    """
    Removes a portion of the worst solutions in our population
    """
    def depredation(self):
        if self.Pd == 1:
            self.full_depredation()
        else:
            # Calculate the number of affected corals
            amount = int(len(self.population)*self.Fd)

            # Select the worse individuals in the grid
            fitness_values = np.array([coral.get_fitness() for coral in self.population])
            affected_corals = list(np.argsort(fitness_values))[:amount]

            # Set a 'dead' flag in the affected corals with a small probability
            alive_count = len(self.population)

            for i in affected_corals:

                # Ensure there are at least 2 individuals in the population
                if alive_count <= 2:
                    break
                
                # Kill the indiviual with probability Pd
                dies = random.random() <= self.Pd
                self.population[i].is_dead = dies
                if dies:
                    alive_count -= 1

            # Remove the dead corals from the population
            self.population = list(filter(lambda c: not c.is_dead, self.population))

    def full_depredation(self):
        # Calculate the number of affected corals
        amount = int(len(self.population)*self.Fd)

        # Select the worse individuals in the grid
        fitness_values = np.array([coral.get_fitness() for coral in self.population])
        affected_corals = list(np.argsort(fitness_values))[:amount]

        # Remove all the individuals chosen
        self.population = [self.population[i] for i in range(len(self.population)) if i not in affected_corals] 

    """
    Makes sure that we calculate the list of solution vectors only once
    """
    def update_identifier_list(self):
        if not self.updated:
            self.identifier_list = [i.solution for i in self.population]
            self.updated = True

    """
    Eliminates duplicate solutions from our population
    """
    def extreme_depredation(self, tol=0):
        # Get a list of the vectors of each individual in the population
        self.update_identifier_list()

        # Store the individuals with vectors repeated more than K times
        repeated_idx = []
        for idx, val in enumerate(self.identifier_list):
            if np.count_nonzero((np.isclose(val,x,tol)).all() for x in self.identifier_list[:idx]) > self.K:
                repeated_idx.append(idx)
        
        # Remove the individuals selected in the previous step
        self.population = [val for idx, val in enumerate(self.population) if idx not in repeated_idx]
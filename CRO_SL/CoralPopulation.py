import random
import numpy as np
from numba import jit
import math

"""
Individual that holds a tentative solution with 
its fitness.
"""
class Coral:
    """
    Constructor
    """
    def __init__(self, solution, objfunc, operator=None):
        self.solution = solution
        
        self.fitness_calculated = False
        self.fitness = 0

        self.is_dead = False

        self.objfunc = objfunc

        self.operator = operator

    def get_fitness(self):
        if not self.fitness_calculated:
            self.fitness = self.objfunc.fitness(self.solution)
            self.fitness_calculated = True
        return self.fitness
    
    def set_operator(self, operator):
        self.operator = operator
    
    def reproduce(self, population):
        new_solution = self.operator.evolve(self, population, self.objfunc)
        new_solution = self.objfunc.check_bounds(new_solution)
        return Coral(new_solution, self.objfunc, self.operator)


"""
Population of corals
"""
class CoralPopulation:    
    """
    Constructor of the Coral Population class
    """
    def __init__(self, objfunc, operators, params, population=None):
        # Hyperparameters of the algorithm
        self.size = params["ReefSize"]
        self.rho = params["rho"]
        self.Fb = params["Fb"]
        self.Fd = params["Fd"]
        self.k = params["k"]
        self.K = params["K"]
        self.Pd = params["Pd"]
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
        self.operators = operators

        # Population initialization
        if population is None:
                self.population = []

        # Operator data structures
        self.operator_list = [i%len(operators) for i in range(self.size)]
        self.operator_weight = [1]*len(operators)

        # Dynamic data structures
        self.operator_data = [[] for i in operators]
        if self.dyn_method == "success":
            for idx, _ in enumerate(self.operator_data):
                self.operator_data[idx].append(0)
            self.larva_count = [0 for i in operators]
        elif self.dyn_method == "diff":
            self.operator_metric_prev = [0]*len(operators)
        self.operator_w_history = []
        self.subs_steps = 0
        self.operator_metric = [0]*len(operators)
        self.operator_history = []
        


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
        amount = int(self.size*self.rho)
        self.population = []
        for i in range(amount):
            operator_idx = self.operator_list[i]
            new_coral = Coral(self.objfunc.random_solution(), self.objfunc, self.operators[operator_idx])
            self.population.append(new_coral)
    
    """
    Evaluates the operators using a given metric
    """
    def evaluate_operators(self):
        for idx, s_data in enumerate(self.operator_data):
            #if self.dyn_method:
            #    self.operator_data = self.operator_data.copy()

            if self.dyn_method == "success":
                if self.larva_count[idx] > 0:
                    self.operator_metric[idx] = s_data[0]/self.larva_count[idx]
                else:
                    self.operator_metric[idx] = 0

                # Reset data for nex iteration
                self.operator_data[idx] = [0]
                self.larva_count[idx] = 0
            elif self.dyn_method == "fitness" or self.dyn_method == "diff":
                if len(s_data) > 0:
                    if self.dyn_metric == "best":
                        self.operator_metric[idx] = max(s_data)
                    elif self.dyn_metric == "avg":
                        self.operator_metric[idx] = sum(s_data)/len(s_data)
                    elif self.dyn_metric == "med":
                        if len(s_data) % 2 == 0:
                            self.operator_metric[idx] = s_data[len(s_data)//2-1]+s_data[len(s_data)//2]
                        else:
                            self.operator_metric[idx] = s_data[len(s_data)//2]
                    elif self.dyn_metric == "worse":
                        self.operator_metric[idx] = min(s_data)
                else:
                    self.operator_metric[idx] = 0
                
                # Reset data for nex iteration
                self.operator_data[idx] = []
            
            if self.dyn_method == "diff":
                aux = self.operator_metric[idx]
                self.operator_metric[idx] = self.operator_metric[idx] - self.operator_metric_prev[idx]
                self.operator_metric_prev[idx] = aux
        
        

    """
    Evolves the population using the corresponding operators
    """
    def evolve_with_operators(self):
        larvae = []
        if self.group_subs:
            # Divide the population based on their operator type
            operator_groups = [[] for i in self.operators]
            for i, idx in enumerate(self.operator_list):
                if i < len(self.population):
                    operator_groups[idx].append(self.population[i])
            
            
            # Reproduce the corals of each group
            for i, coral_group in enumerate(operator_groups):
                # Restart fitness record if there are corals in this operator

                for coral in coral_group:
                    # Generate new coral
                    if random.random() <= self.Fb:
                        new_coral = coral.reproduce(coral_group)
                        
                        # Get data of the current operator
                        if self.dyn_method == "fitness" or self.dyn_method == "diff":
                            self.operator_data[i].append(new_coral.get_fitness())
                    else:
                        new_coral = Coral(self.objfunc.random_solution(), self.objfunc)

                    # Add larva to the list of larvae
                    larvae.append(new_coral)
        else:
            for idx, coral in enumerate(self.population):
                # Generate new coral
                if random.random() <= self.Fb:
                    new_coral = coral.reproduce(self.population)
                    
                    # Get operator index
                    s_names = [i.evolution_method for i in self.operators]
                    s_idx = s_names.index(coral.operator.evolution_method)

                    # Get data of the current operator
                    if self.dyn_method == "fitness" or self.dyn_method == "diff":
                        self.operator_data[s_idx].append(new_coral.get_fitness())
                else:
                    new_coral = Coral(self.objfunc.random_solution(), self.objfunc)

                # Add larva to the list of larvae
                larvae.append(new_coral)
        return larvae

    """
    Converts the evaluation values of the operators to a probability distribution
    """
    def operator_probability(self, values):
        # Normalization
        weight = np.array(values)
        if weight.sum() != 0:
            weight = weight/np.abs(weight.sum())
        
        # softmax to convert to a probability distribution
        exp_vec = np.exp(weight)
        amplified_vec = exp_vec**(1/self.prob_amp)
        
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
    Generates the assignment of the operators
    """
    def generate_operators(self, progress=0):
        if len(self.operators) == 1:
            self.operator_list = [0] * self.size
        else:
            if progress > (1/self.dyn_steps)*self.subs_steps:
                self.subs_steps += 1

                n_operators = len(self.operators)
                self.evaluate_operators()
                if self.dynamic:
                    # Assign the probability of each operator
                    self.operator_weight = self.operator_probability(self.operator_metric)
                    self.operator_w_history.append(self.operator_weight)
                
                # Choose each operator with the weights chosen
                self.operator_list = random.choices(range(n_operators), 
                                                    weights=self.operator_weight, k=self.size)

                # Assign the operator to each coral
                for idx, coral in enumerate(self.population):
                    operator_idx = self.operator_list[idx]
                    coral.set_operator(self.operators[operator_idx])

                self.operator_history.append(np.array(self.operator_metric))

    """
    Inserts solutions into our reef with some conditions
    """
    def larvae_setting(self, larvae_list):
        s_names = [i.evolution_method for i in self.operators]
        for larva in larvae_list:
            attempts_left = self.k
            setted = False
            idx = -1

            # Try to settle 
            while attempts_left > 0 and not setted:
                # Choose a random position
                idx = random.randrange(0, self.size)

                # If it's empty settle in there if it's not the case,
                # try to replace the coral in that position
                if idx >= len(self.population):
                    setted = True
                    self.population.append(larva)
                elif setted := (larva.get_fitness() > self.population[idx].fitness):
                    self.population[idx] = larva

                attempts_left -= 1
            
            if larva.operator is not None:
                s_idx = s_names.index(larva.operator.evolution_method)
                if self.dyn_method == "success":
                    self.larva_count[s_idx] += 1

            # Assign operator to the setted coral
            if setted:
                # Get operator index
                if self.dyn_method == "success" and larva.operator is not None:
                    self.operator_data[s_idx][0] += 1

                operator_idx = self.operator_list[idx]
                larva.set_operator(self.operators[operator_idx])
                
        
    """
    Inserts a new random element into the population 
    """
    def budding(self):
        if len(self.population) <= self.size*0.2:
            n_corals = int(self.size*self.Fb)

            duplicates = sorted(self.population, reverse=True, key = lambda c: c.get_fitness())[:n_corals]

            self.larvae_setting(duplicates)
    
    """
    Removes a portion of the worst solutions in our population
    """
    def depredation(self):
        if self.Fd == 1:
            self.full_depredation()
            return
        
        # Calculate the number of affected corals
        amount = int(self.size*self.Pd)

        # Extract the worse 'n_corals' in the grid
        fitness_values = np.array([coral.get_fitness() for coral in self.population])
        affected_corals = list(np.argsort(fitness_values))[:amount]

        # Set a 'dead' flag in the affected corals with a small probability
        alive_count = len(self.population)
        for i in affected_corals:
            if alive_count <= 2:
                break
            dies = random.random() <= self.Fd
            self.population[i].is_dead = dies
            if dies:
                alive_count -= 1

        # Remove the dead corals from the population
        self.population = list(filter(lambda c: not c.is_dead, self.population))

    def full_depredation(self):
        # Calculate the number of affected corals
        amount = int(self.size*self.Pd)

        fitness_values = np.array([coral.get_fitness() for coral in self.population])
        affected_corals = list(np.argsort(fitness_values))[:amount]

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
        self.update_identifier_list()

        repeated_idx = []
        for idx, val in enumerate(self.identifier_list):
            if np.count_nonzero((np.isclose(val,x,tol)).all() for x in self.identifier_list[:idx]) > self.K:
                repeated_idx.append(idx)

        self.population = [val for idx, val in enumerate(self.population) if idx not in repeated_idx]
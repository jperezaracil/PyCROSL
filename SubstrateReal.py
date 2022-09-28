import numpy as np
import random
from Substrate import *


"""
Substrate class that has continuous mutation and cross methods
"""
class SubstrateReal(Substrate):
    def __init__(self, evolution_method, params = None):
        self.evolution_method = evolution_method
        super().__init__(self.evolution_method, params)
    
    """
    Evolves a solution with a different strategy depending on the type of substrate
    """
    def evolve(self, solution, population, objfunc):
        result = None
        others = [i for i in population if i != solution]
        if len(others) > 1:
            solution2 = random.choice(others)
        else:
            solution2 = solution
        
        if self.evolution_method == "1point":
            result = cross1p(solution.solution.copy(), solution2.solution.copy())
        elif self.evolution_method == "2point":
            result = cross2p(solution.solution.copy(), solution2.solution.copy())
        elif self.evolution_method == "Multipoint":
            result = crossMp(solution.solution.copy(), solution2.solution.copy())
        elif self.evolution_method == "Gauss":
            result = gaussian(solution.solution.copy(), self.params["F"])
        elif self.evolution_method == "BLXalpha":
            result = blxalpha(solution.solution.copy(), solution2.solution.copy(), self.params["F"])
        elif self.evolution_method == "SBX":
            result = sbx(solution.solution.copy(), solution2.solution.copy(), self.params["F"])
        elif self.evolution_method == "Perm":
            result = permutation(solution.solution.copy(), self.params["F"])
        elif self.evolution_method == "DE/rand/1":
            result = DEBest1(solution.solution.copy(), others, self.params["F"], self.params["Pr"])
        elif self.evolution_method == "DE/best/1":
            result = DERand1(solution.solution.copy(), others, self.params["F"], self.params["Pr"])
        elif self.evolution_method == "DE/rand/2":
            result = DEBest2(solution.solution.copy(), others, self.params["F"], self.params["Pr"])
        elif self.evolution_method == "DE/best/2":
            result = DERand2(solution.solution.copy(), others, self.params["F"], self.params["Pr"])
        elif self.evolution_method == "DE/current-to-rand/1":
            result = DECurrentToBest1(solution.solution.copy(), others, self.params["F"], self.params["Pr"])
        elif self.evolution_method == "DE/current-to-best/1":
            result = DECurrentToRand1(solution.solution.copy(), others, self.params["F"], self.params["Pr"])
        elif self.evolution_method == "SA":
            result = sim_annealing(solution, self.params["F"], objfunc, self.params["temp_ch"], self.params["iter"])
        elif self.evolution_method == "HS":
            result = harmony_search(solution.solution.copy(), population, self.params["F"], self.params["Pr"], 0.4)
        elif self.evolution_method == "Rand":
            result = random_replace(solution.solution.copy())
        else:
            print(f"Error: evolution method \"{self.evolution_method}\" not defined")
            exit(1)
            
        
        return result

## Mutation and recombination methods
def random_replace(solution):
    return solution.max()*np.random.random(solution.shape)-2*solution.min()

def gaussian(solution, strength):
    return solution + np.random.normal(0,strength,solution.shape)

def permutation(solution, strength):
    mask = np.random.random(solution.shape) < strength
    if np.count_nonzero(mask==1) < 2:
        mask[random.sample(range(mask.size), 2)] = 1 
    np.random.shuffle(solution[mask])
    return solution

def cross1p(solution1, solution2):
    cross_point = random.randrange(0, solution1.size)
    return np.hstack([solution1[:cross_point], solution2[cross_point:]])

def cross2p(solution1, solution2):
    cross_point1 = random.randrange(0, solution1.size-2)
    cross_point2 = random.randrange(cross_point1, solution1.size)
    return np.hstack([solution1[:cross_point1], solution2[cross_point1:cross_point2], solution1[cross_point2:]])

def crossMp(solution1, solution2):
    vector = 1*(np.random.rand(solution1.size) > 0.5)
    aux = np.copy(solution1)
    aux[vector==1] = solution2[vector==1]
    return aux

def blxalpha(solution1, solution2, alpha):
    return alpha*solution1 + (1-alpha)*solution2

def sbx(solution1, solution2, strength):
    beta = np.zeros(solution1.shape)
    u = np.random.random(solution1.shape)
    for idx, val in enumerate(u):
        if val <= 0.5:
            beta[idx] = (2*val)**(1/(strength+1))
        else:
            beta[idx] = (0.5*(1-val))**(1/(strength+1))
    
    sign = random.choice([-1,1])
    return 0.5*(solution1+solution2) + sign*0.5*beta*(solution1-solution2)

def sim_annealing(solution, strength, objfunc, temp_changes, iter):
    p0, pf = (0.1, 7)

    alpha = 0.99
    best_fit = solution.get_fitness()
    prev_solution = solution.solution

    temp_init = temp = 100
    temp_fin = alpha**temp_changes * temp_init
    while temp >= temp_fin:
        for j in range(iter):
            solution_new = gaussian(solution.solution, strength)
            new_fit = objfunc.fitness(solution_new)
            y = ((pf-p0)/(temp_fin-temp_init))*(temp-temp_init) + p0 

            p = np.exp(-y)
            if new_fit > best_fit or random.random() < p:
                prev_solution = solution_new
                best_fit = new_fit
        temp = temp*alpha
    return solution_new

def harmony_search(solution, population, strength, HMCR, PAR):
    new_solution = np.zeros(solution.shape)
    popul_matrix = np.vstack([i.solution for i in population])
    popul_mean = popul_matrix.mean(axis=0)
    popul_std = popul_matrix.std(axis=0)
    for i in range(solution.size):
        if random.random() < HMCR:
            new_solution[i] = random.choice(population).solution[i]
            if random.random() <= PAR:
                new_solution[i] += np.random.normal(0,strength)
        else:
            new_solution[i] = np.random.normal(popul_mean[i], popul_std[i])
    return new_solution


def DERand1(solution, population, F, CR):
    if len(population) > 3:
        r1, r2, r3 = random.sample(population, 3)

        v = r1.solution + F*(r2.solution-r3.solution)
        mask = np.random.random(solution.shape) <= CR
        solution[mask] = v[mask]
    return solution

def DEBest1(solution, population, F, CR):
    if len(population) > 3:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2 = random.sample(population, 2)

        v = best.solution + F*(r1.solution-r2.solution)
        mask = np.random.random(solution.shape) <= CR
        solution[mask] = v[mask]
    return solution

def DERand2(solution, population, F, CR):
    if len(population) > 5:
        r1, r2, r3, r4, r5 = random.sample(population, 5)

        v = r1.solution + F*(r2.solution-r3.solution) + F*(r4.solution-r5.solution)
        mask = np.random.random(solution.shape) <= CR
        solution[mask] = v[mask]
    return solution

def DEBest2(solution, population, F, CR):
    if len(population) > 5:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2, r3, r4 = random.sample(population, 4)

        v = best.solution + F*(r1.solution-r2.solution) + F*(r3.solution-r4.solution)
        mask = np.random.random(solution.shape) <= CR
        solution[mask] = v[mask]
    return solution

def DECurrentToBest1(solution, population, F, CR):
    if len(population) > 3:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2 = random.sample(population, 2)

        v = solution + F*(best.solution-solution) + F*(r1.solution-r2.solution)
        mask = np.random.random(solution.shape) <= CR
        solution[mask] = v[mask]
    return solution

def DECurrentToRand1(solution, population, F, CR):
    if len(population) > 3:
        r1, r2, r3 = random.sample(population, 3)

        v = solution + np.random.random()*(r1.solution-solution) + F*(r2.solution-r3.solution)
        mask = np.random.random(solution.shape) <= CR
        solution[mask] = v[mask]
    return solution
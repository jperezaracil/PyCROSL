import numpy as np
import random
from Operator import *

"""
Operator class that has discrete mutation and cross methods
"""
class OperatorInt(Operator):
    def __init__(self, evolution_method, params = None):
        self.evolution_method = evolution_method
        super().__init__(self.evolution_method, params)
    
    """
    Applies a mutation method depending on the type of operator
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
        #elif self.evolution_method == "BLXalpha":
        #    result = bxalpha(solution.solution.copy(), solution2.solution.copy())
        elif self.evolution_method == "SBX":
            result = sbx(solution.solution.copy(), solution2.solution.copy(), self.params["F"])
        elif self.evolution_method == "Perm":
            result = permutation(solution.solution.copy(), self.params["F"])
        elif self.evolution_method == "Xor":
            result = xorMask(solution.solution.copy(), self.params["F"])
        elif self.evolution_method == "DGauss":
            result = discreteGaussian(solution.solution.copy(), self.params["F"])
        elif self.evolution_method == "AddOne":
            result = addOne(solution.solution.copy(), self.params["F"])
        elif self.evolution_method == "DE/rand/1":
            result = DERand1(solution.solution.copy(), others, self.params["F"], self.params["Pr"])
        elif self.evolution_method == "DE/best/1":
            result = DEBest1(solution.solution.copy(), others, self.params["F"], self.params["Pr"])
        elif self.evolution_method == "DE/rand/2":
            result = DERand2(solution.solution.copy(), others, self.params["F"], self.params["Pr"])
        elif self.evolution_method == "DE/best/2":
            result = DEBest2(solution.solution.copy(), others, self.params["F"], self.params["Pr"])
        elif self.evolution_method == "DE/current-to-rand/1":
            result = DECurrentToRand1(solution.solution.copy(), others, self.params["F"], self.params["Pr"])
        elif self.evolution_method == "DE/current-to-best/1":
            result = DECurrentToBest1(solution.solution.copy(), others, self.params["F"], self.params["Pr"])
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
    return np.random.randint(solution.max(), solution.min(), size=solution.shape)

def discreteGaussian(solution, strength):
    return (solution + np.random.normal(0,strength,solution.shape)).astype(np.int32)

def addOne(solution, strength):
    increase = np.random.choice([-1,0,1], size=solution.size, p=[strength/2, 1-strength, strength/2])
    return (solution + increase).astype(np.int32)

def xorMask(solution, strength):
    if min(solution) == 0 and max(solution) == 1:
        vector = np.random.random(solution.shape) < strength
    else:
        mask = (np.random.random(solution.shape) < strength).astype(np.int32)
        vector = np.random.randint(1, 0xFF, size=solution.shape) * mask
    return solution ^ vector

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

def blxalpha(solution1, solution2, alpha=0.7):
    return (alpha*solution1 + (1-alpha)*solution2).astype(np.int32)

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

def sim_annealing(solution, strength, objfunc, temp_changes=10, iter=5):
    p0, pf = (0.1, 7)

    alpha = 0.99
    best_fit = solution.get_fitness()
    prev_solution = solution.solution

    temp_init = temp = 100
    temp_fin = alpha**temp_changes * temp_init
    while temp >= temp_fin:
        for j in range(iter):
            solution_new = discreteGaussian(solution.solution, strength)
            new_fit = objfunc.fitness(solution_new)
            y = ((pf-p0)/(temp_fin-temp_init))*(temp-temp_init) + p0 

            p = np.exp(-y)
            if new_fit > best_fit or random.random() < p:
                prev_solution = solution_new
                best_fit = new_fit
        temp = temp*alpha
    return solution_new

def harmony_search(solution, population, strength, HMCR=0.8, PAR=0.3):
    new_solution = np.zeros(solution.shape)
    popul_matrix = np.vstack([i.solution for i in population])
    popul_mean = popul_matrix.mean(axis=0)
    popul_std = popul_matrix.std(axis=0)
    for i in range(solution.size):
        if random.random() < HMCR:
            new_solution[i] = random.choice(population).solution[i]
            if random.random() <= PAR:
                new_solution[i] += int(np.random.normal(0,strength))
        else:
            new_solution[i] = int(np.random.normal(popul_mean[i], popul_std[i]))
    return new_solution

def DERand1(solution, population, F, CR):
    if len(population) > 3:
        r1, r2, r3 = random.sample(population, 3)

        v = (r1.solution + F*(r2.solution-r3.solution)).astype(np.int32)
        mask = np.random.random(solution.shape) <= CR
        solution[mask] = v[mask]
    return solution

def DEBest1(solution, population, F, CR):
    if len(population) > 3:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2 = random.sample(population, 2)

        v = (best.solution + F*(r1.solution-r2.solution)).astype(np.int32)
        mask = np.random.random(solution.shape) <= CR
        solution[mask] = v[mask]
    return solution

def DERand2(solution, population, F, CR):
    if len(population) > 5:
        r1, r2, r3, r4, r5 = random.sample(population, 5)

        v = (r1.solution + F*(r2.solution-r3.solution) + F*(r4.solution-r5.solution)).astype(np.int32)
        mask = np.random.random(solution.shape) <= CR
        solution[mask] = v[mask]
    return solution

def DEBest2(solution, population, F, CR):
    if len(population) > 5:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2, r3, r4 = random.sample(population, 4)

        v = (best.solution + F*(r1.solution-r2.solution) + F*(r3.solution-r4.solution)).astype(np.int32)
        mask = np.random.random(solution.shape) <= CR
        solution[mask] = v[mask]
    return solution

def DECurrentToBest1(solution, population, F, CR):
    if len(population) > 3:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2 = random.sample(population, 2)

        v = (solution + F*(best.solution-solution) + F*(r1.solution-r2.solution)).astype(np.int32)
        mask = np.random.random(solution.shape) <= CR
        solution[mask] = v[mask]
    return solution

def DECurrentToRand1(solution, population, F, CR):
    if len(population) > 3:
        r1, r2, r3 = random.sample(population, 3)

        v = (solution + np.random.random()*(r1.solution-solution) + F*(r2.solution-r3.solution)).astype(np.int32)
        mask = np.random.random(solution.shape) <= CR
        solution[mask] = v[mask]
    return solution

# https://www.frontiersin.org/articles/10.3389/fbuil.2020.00102/full
# https://www.researchgate.net/publication/221008091_Modified_SBX_and_adaptive_mutation_for_real_world_single_objective_optimization
# https://stackoverflow.com/questions/38952879/blx-alpha-crossover-what-approach-is-the-right-one
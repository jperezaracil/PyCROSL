import math
import numpy as np
import scipy as sp
import scipy.stats
import random

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

def replace(solution, population, method, strength):
    popul_matrix = np.vstack([i.solution for i in population])
    mean = popul_matrix.mean(axis=0)
    std = popul_matrix.std(axis=0)*strength
    if method == "Laplace":
        return sp.stats.laplace.rvs(mean, std,solution.shape)
    elif method == "Cauchy":
        return sp.stats.cauchy.rvs(mean, std,solution.shape)
    elif method == "Gauss":
        return np.random.normal(mean, std,solution.shape)
    elif method == "Uniform":
        return solution.max()*np.random.random(solution.shape)-2*solution.min()
        #return popul_matrix.max(axis=0)*np.random.random(solution.shape)-2*popul_matrix.min(axis=0)

def laplace(solution, strength):
    return solution + sp.stats.laplace.rvs(0,strength,solution.shape)

def cauchy(solution, strength):
    return solution + sp.stats.cauchy.rvs(0,strength,solution.shape)

def gaussian(solution, strength):
    return solution + np.random.normal(0, strength,solution.shape)

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

def multiCross(solution1, population, n_ind):
    if len(population) < n_ind:
        n_ind = len(population)-1
    vector = np.random.randint(n_ind, size=solution1.size)
    parents = random.sample(population, n_ind-1)
    aux = np.copy(solution1)
    for i in range(1, n_ind-1):
        aux[vector==i] = parents[i].solution[vector==i]
    return aux

def blxalpha(solution1, solution2, alpha):
    alpha *= np.random.random()
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

    temp_init = temp = 100
    temp_fin = alpha**temp_changes * temp_init
    while temp >= temp_fin:
        for j in range(iter):
            solution_new = gaussian(solution.solution, strength)
            new_fit = objfunc.fitness(solution_new)
            y = ((pf-p0)/(temp_fin-temp_init))*(temp-temp_init) + p0 

            p = np.exp(-y)
            if new_fit > best_fit or random.random() < p:
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

def DECurrentToPBest1(solution, population, F, CR, p=0.11):
    if len(population) > 3:
        fitness = [i.fitness for i in population]
        pbest_idx = random.choice(np.argsort(fitness)[:math.ceil(len(population)*p)])
        pbest = population[pbest_idx]
        r1, r2 = random.sample(population, 2)

        v = solution + F*(pbest.solution-solution) + F*(r1.solution-r2.solution)
        mask = np.random.random(solution.shape) <= CR
        solution[mask] = v[mask]
    return solution

def dummy_op(solution, scale=1000):
    return np.ones(solution.shape)*scale
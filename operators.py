import math
import random

import numpy as np
import scipy as sp
import scipy.stats


## Mutation and recombination methods
def xorMask(solution, strength, mode="byte"):
    if mode == "bin":
        vector = np.random.random(solution.shape) < strength
    elif mode == "byte":
        mask = (np.random.random(solution.shape) < strength).astype(np.int32)
        vector = np.random.randint(1, 0xFF, size=solution.shape) * mask
    elif mode == "int":
        mask = (np.random.random(solution.shape) < strength).astype(np.int32)
        vector = np.random.randint(1, 0xFFFF, size=solution.shape) * mask
    return solution ^ vector

def permutation(solution, strength):
    mask = np.random.random(solution.shape) < strength
    if np.count_nonzero(mask==1) < 2:
        mask[random.sample(range(mask.size), 2)] = 1 
    np.random.shuffle(solution[mask])
    return solution

def mutate_rand(solution, population, method, strength, prop):
    amount = max(1, int(solution.size*prop))
    idx = random.sample(range(solution.size), amount)

    popul_matrix = np.vstack([i.solution for i in population])
    mean = popul_matrix.mean(axis=0)[idx]
    std = (popul_matrix.std(axis=0)[idx] + 1e-6)*strength # ensure there will be some standard deviation

    random_vec = None
    if method == "Laplace":
        random_vec = sp.stats.laplace.rvs(mean, std, size=amount)
    elif method == "Cauchy":
        random_vec = sp.stats.cauchy.rvs(mean, std, size=amount)
    elif method == "Gauss":
        random_vec = sp.stats.norm.rvs(mean, std, size=amount)
    elif method == "Uniform":
        random_vec = solution.max()*np.random.random(size=amount)-2*solution.min()
    elif method == "Poisson":
        random_vec = sp.stats.poisson.rvs(mean, size=amount)
    
    
    solution[idx] = random_vec
    return solution

def replace(solution, population, method, strength):
    popul_matrix = np.vstack([i.solution for i in population])
    mean = popul_matrix.mean(axis=0)
    std = popul_matrix.std(axis=0)*strength
    if method == "Laplace":
        return sp.stats.laplace.rvs(mean, std, size=solution.shape)
    elif method == "Cauchy":
        return sp.stats.cauchy.rvs(mean, std, size=solution.shape)
    elif method == "Gauss":
        return np.random.normal(mean, std, size=solution.size)
    elif method == "Uniform":
        return solution.max()*np.random.random(size=solution.shape)-2*solution.min()
    elif method == "Poisson":
        return sp.stats.poisson.rvs(mean, size=solution.shape)

def laplace(solution, strength):
    return solution + sp.stats.laplace.rvs(0, strength, size=solution.shape)

def cauchy(solution, strength):
    return solution + sp.stats.cauchy.rvs(0, strength, size=solution.shape)

def gaussian(solution, strength):
    return solution + np.random.normal(0, strength, size=solution.shape)

def uniform(solution, strength):
    return solution + np.random.uniform(-strength, strength, size=solution.shape)

def poisson(solution, mu):
    return solution + sp.stats.poisson.rvs(mu, size=solution.shape)

"""
-Distribución zeta
-Distribución hipergeométrica
-Distribución geomética
-Distribución de Boltzman
-Distribución de Pascal (binomial negativa)
"""

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
    if n_ind <= len(population):
        n_ind = len(population)-1
        vector = np.random.randint(n_ind, size=solution1.size)
        parents = random.sample(population, n_ind)
        aux = np.copy(solution1)
        for i in range(1, n_ind-1):
            aux[vector==i] = parents[i].solution[vector==i]
        return aux
    else:
        return solution1

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

def firefly(solution, population, objfunc, alpha_0, beta_0, delta, gamma):
    sol_range = objfunc.sup_lim - objfunc.inf_lim
    n_dim = solution.solution.size
    new_solution = solution.solution.copy()
    for idx, ind in enumerate(population):
        if solution.get_fitness() < ind.get_fitness():
            r = np.linalg.norm(solution.solution - ind.solution)
            alpha = alpha_0 * delta ** idx
            beta = beta_0 * np.exp(-gamma*(r/(sol_range*np.sqrt(n_dim)))**2)
            new_solution = new_solution + beta*(ind.solution-new_solution) + alpha * sol_range * random.random()-0.5
            new_solution = objfunc.check_bounds(new_solution)
    
    return new_solution

"""
Only for testing, not useful for real applications
"""
def dummy_op(solution, scale=1000):
    return np.ones(solution.shape)*scale
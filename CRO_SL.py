import numpy as np
from matplotlib import pyplot as plt 
from CoralPopulation import CoralPopulation
from ExampleObj import ExampleObj
from Substrate import *
import time

"""
Coral reef optimization with substrate layers

Parameters:
    Ngen: number of generations
    ReefSize: maximum number of corals in the reef
    rho: percentage of initial ocupation of the reef
    Fb: broadcast spawning proportion
    Fa: asexual reproduction proportion
    Fd: depredation probability
    Pd: depredation proportion
    k: maximum trys for larva setting
    K: maximum amount of equal corals
"""
class CRO_SL:
    def __init__(self, objfunc, substrates, params):
        # Hyperparameters of the algorithm
        self.ReefSize = params["ReefSize"]
        self.rho = params["rho"]
        self.Fb = params["Fb"]
        self.Fa = params["Fa"]
        self.Fd = params["Fd"]
        self.k = params["k"]
        self.K = params["K"]
        self.Pd = params["Pd"]

        # Stopping conditions
        self.stop_cond = params["stop_cond"]
        self.Ngen = params["Ngen"]
        self.Neval = params["Neval"]
        self.time_limit = params["time_limit"]
        self.fit_target = params["fit_target"]

        # Data structures of the algorithm
        self.objfunc = objfunc
        self.substrates = substrates
        self.population = CoralPopulation(self.ReefSize, self.objfunc, self.substrates)
        
        # Metrics
        self.history = []
        self.best_fitness = 0
        self.time_spent = 0
    
    def init_population(self):
        self.population.generate_random(self.rho)
    
    def broadcast_spawning(self):
        return self.population.broadcast_spawning(self.Fb)
    
    def brooding(self, corals_chosen):
        return self.population.brooding(corals_chosen, 1-self.Fb)
    
    def larvae_setting(self, larvae):
        self.population.larvae_setting(larvae, self.k)
    
    def budding(self):
        self.population.budding(self.Fa, self.k)
    
    def depredation(self):
        self.population.depredation(self.Pd, self.Fd)

    def step(self, depredate=True):
        self.population.generate_substrates()

        larvae_broadcast, corals_chosen = self.broadcast_spawning()

        larvae_brooding = self.brooding(corals_chosen)

        larvae = larvae_broadcast + larvae_brooding
        self.larvae_setting(larvae)

        self.budding()

        if depredate:
            self.depredation()

        _, best_fitness = self.population.best_solution()
        self.history.append(best_fitness)
    
    def stopping_condition(self, gen, time_start):
        stop = True
        if self.stop_cond == "neval":
            stop = self.population.fitness_count >= self.fitness_count
        elif self.stop_cond == "ngen":
            stop = gen >= self.Ngen
        elif self.stop_cond == "time":
            stop = time.time()-time_start >= self.time_limit
        elif self.stop_cond == "fit_target":
            stop = self.population.best_solution()[1] >= self.fit_target
        
        return stop

    def optimize(self):
        self.init_population()
        self.step(depredate=False)
        gen = 0
        time_start = time.time()
        while not self.stopping_condition(gen, time_start):
            self.step()
            gen += 1
        self.time_spent = time.time() - time_start
        return self.population.best_solution()
    
    def display_report(self):
        print("Number of generations:", len(self.history))
        print("Time spent: ", round(self.time_spent, 5), "s", sep="")
        print("Number of fitness evaluations:", self.population.fitness_count)
        print("Best fitness:", self.population.best_solution()[1])
        plt.plot(self.history)
        plt.xlabel("generations")
        plt.ylabel("fitness")
        plt.title("CRO_SL")
        plt.show()
        
def main():
    substrates = [SubstrateInt("Xor", "1point"), SubstrateInt("Xor", "2point"), SubstrateInt("Xor", "Multipoint"),
     SubstrateInt("Perm", "1point"), SubstrateInt("Perm", "2point"), SubstrateInt("Perm", "Multipoint")]
    
    params = {
        "ReefSize": 500,
        "rho": 0.5,
        "Fb": 0.1,
        "Fa": 0.01,
        "Fd": 0.14,
        "k": 3,
        "Pd": 0.6,
        "K": 20,

        "stop_cond": "time",
        "time_limit": 3.0,
        "Ngen": 500,
        "Neval":  4000,
        "fit_target": 600
    }
    c = CRO_SL(ExampleObj(1000), substrates, params)
    ind, fit = c.optimize()
    print(ind)
    c.display_report()
    

if __name__ == "__main__":
    main()

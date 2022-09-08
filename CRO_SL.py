import numpy as np
from matplotlib import pyplot as plt 
from CoralPopulation import CoralPopulation
from ExampleObj import ExampleObj
from Substrate import Substrate

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
    def __init__(self, objfunc, substrates, Ngen=500, ReefSize=300,
                rho=0.6, Fb=0.9, Fa=0.01, Fd=0.01, k=3, Pd=0.1, K=20):
        self.objfunc = objfunc
        self.substrates = substrates
        self.population = CoralPopulation(ReefSize, self.objfunc, self.substrates)
        self.history = []
        

        self.Ngen = Ngen
        self.rho = rho
        self.Fb = Fb
        self.Fa = Fa
        self.Fd = Fd
        self.k = k
        self.Pd = Pd
        self.K = K
    
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
        larvae_broadcast, corals_chosen = self.broadcast_spawning()

        larvae_brooding = self.brooding(corals_chosen)

        larvae = larvae_broadcast + larvae_brooding
        self.larvae_setting(larvae)

        self.budding()

        if depredate:
            self.depredation()

        _, best_fitness = self.population.best_solution()
        self.history.append(best_fitness)
    
    def optimize(self):
        self.init_population()
        self.step(depredate=False)
        for i in range(self.Ngen):
            self.step()
        return self.population.best_solution()
    
def main():
    substrates = [Substrate("Gauss", "1point"), Substrate("Gauss", "2point")]
    c = CRO_SL(ExampleObj(100), substrates)
    #c.init_population()
    #c.step()
    ind, fit = c.optimize()

    print(ind)
    plt.plot(c.history)
    plt.show()

if __name__ == "__main__":
    main()

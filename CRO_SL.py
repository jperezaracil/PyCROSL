import numpy as np
from matplotlib import pyplot as plt 
from CoralPopulation import CoralPopulation
from TestFunctions import *
from Substrate import *
from SubstrateInt import *
from SubstrateReal import *
import time

"""
Coral reef optimization with substrate layers

Parameters:
    Ngen: number of generations
    ReefSize: maximum number of corals in the reef
    rho: percentage of initial ocupation of the reef
    mut_str: strength of the mutations
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
        self.mut_str = params["mut_str"]
        self.Fb = params["Fb"]
        self.Fa = params["Fa"]
        self.Fd = params["Fd"]
        self.k = params["k"]
        self.K = params["K"]
        self.Pd = params["Pd"]
        
        # Maximization or Minimization
        self.opt = params["opt"]

        # Verbose parameters
        self.verbose = params["verbose"]
        self.v_timer = params["v_timer"]

        # Stopping conditions
        self.stop_cond = params["stop_cond"]
        self.Ngen = params["Ngen"]
        self.Neval = params["Neval"]
        self.time_limit = params["time_limit"]
        self.fit_target = params["fit_target"]

        # Data structures of the algorithm
        self.objfunc = objfunc
        self.substrates = substrates
        self.population = CoralPopulation(self.ReefSize, self.objfunc, self.substrates, self.opt)
        
        # Metrics
        self.history = []
        self.best_fitness = 0
        self.time_spent = 0
        self.real_time_spent = 0
    
    def init_population(self):
        self.population.generate_random(self.rho)
    
    def broadcast_spawning(self):
        return self.population.broadcast_spawning(self.Fb)
    
    def brooding(self, corals_chosen):
        return self.population.brooding(corals_chosen, self.mut_str)
    
    def evolve_with_substrates(self):
        return self.population.evolve_with_substrates(self.mut_str)
    
    def larvae_setting(self, larvae):
        self.population.larvae_setting(larvae, self.k)
    
    def budding(self):
        self.population.budding(self.Fa, self.k)
    
    def depredation(self):
        self.population.depredation(self.Pd, self.Fd)

    def step(self, depredate=True):
        self.population.generate_substrates()

        larvae = self.evolve_with_substrates()
        
        self.larvae_setting(larvae)
        self.budding()
        if depredate:
            self.depredation()

        _, best_fitness = self.population.best_solution()
        if self.opt == "min":
            best_fitness *= -1
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
            if self.opt == "max":
                stop = self.population.best_solution()[1] >= self.fit_target
            else:
                stop = -self.population.best_solution()[1] <= self.fit_target

        return stop

    def optimize(self):
        gen = 0
        time_start = time.process_time()
        real_time_start = time.time()
        display_timer = time.time()

        self.init_population()
        self.step(depredate=False)
        while not self.stopping_condition(gen, real_time_start):
            self.step()
            gen += 1
            if self.verbose and time.time() - display_timer > self.v_timer:
                self.step_info(gen, display_timer)
                display_timer = time.time()
                
        self.real_time_spent = time.time() - real_time_start
        self.time_spent = time.process_time() - time_start
        return self.population.best_solution()
    
    def step_info(self, gen, start_time):
        print(f"Time Spent {round(time.time() - start_time,2)}s:")
        print(f"\tGeneration: {gen}")
        best_fitness = self.population.best_solution()[1]
        if self.opt == "min":
            best_fitness *= -1
        print(f"\tBest fitness: {best_fitness}")
        print(f"\tEvaluations of fitness: {self.population.fitness_count}\n")

    def display_report(self):
        print("Number of generations:", len(self.history))
        print("Real time spent: ", round(self.real_time_spent, 5), "s", sep="")
        print("CPU time spent: ", round(self.time_spent, 5), "s", sep="")
        print("Number of fitness evaluations:", self.population.fitness_count)
        best_fitness = self.population.best_solution()[1]
        if self.opt == "min":
            best_fitness *= -1
        print("Best fitness:", best_fitness)
        plt.plot(self.history)
        plt.xlabel("generations")
        plt.ylabel("fitness")
        plt.title("CRO_SL")
        plt.show()
        
def main():
    substrates_int = [SubstrateInt("AddOne"), SubstrateInt("DGauss"), SubstrateInt("Perm"),
    SubstrateInt("1point"), SubstrateInt("2point"), SubstrateInt("Multipoint"), SubstrateInt("Xor")]

    substrates_real = [SubstrateReal("Gauss"), SubstrateReal("Perm"), SubstrateReal("1point"),
     SubstrateReal("2point"), SubstrateReal("Multipoint")]
    
    params = {
        "ReefSize": 500,
        "rho": 0.5,
        "mut_str": 0.04,
        "Fb": 0.1,
        "Fa": 0.01,
        "Fd": 0.14,
        "Pd": 0.6,
        "k": 3,
        "K": 20,
        "create_duplicates": True,

        "stop_cond": "time",
        "time_limit": 30.0,
        "Ngen": 100,
        "Neval": 4000,
        "fit_target": 1000,

        "opt": "max",

        "verbose": True,
        "v_timer": 5
    }

    #c = CRO_SL(DiophantineEq(100000, np.random.randint(-100, 100, size=100000), -12), substrates_int, params)
    c = CRO_SL(MaxOnes(1000), substrates_int, params)
    #c = CRO_SL(Sphere(1000), substrates_real, params)
    #c = CRO_SL(MaxOnesReal(1000), substrates_real, params)
    #c = CRO_SL(Test1(1000), substrates_real, params)
    ind, fit = c.optimize()
    print(ind)
    c.display_report()
    

if __name__ == "__main__":
    main()

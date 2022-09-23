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
        self.Fd = params["Fd"]
        self.k = params["k"]
        self.K = params["K"]
        self.Pd = params["Pd"]
        self.dynamic = params["dynamic"]
        
        # Maximization or Minimization
        self.opt = objfunc.opt

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
    
    def evolve_with_substrates(self):
        return self.population.evolve_with_substrates()
    
    def larvae_setting(self, larvae):
        self.population.larvae_setting(larvae, self.k)
    
    #def budding(self):
        #self.population.budding(self.Fa, self.k)
    
    def depredation(self):
        self.population.depredation(self.Pd, self.Fd)

    def step(self, depredate=True, dynamic=True):
        if dynamic:
            self.population.generate_substrates()

        larvae = self.evolve_with_substrates()
        
        self.larvae_setting(larvae)
        
        if depredate:
            self.depredation()

        _, best_fitness = self.population.best_solution()
        if self.opt == "min":
            best_fitness *= -1
        self.history.append(best_fitness)
    
    def stopping_condition(self, gen, time_start):
        stop = True
        if self.stop_cond == "neval":
            stop = self.objfunc.counter >= self.fitness_count
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
        self.step(depredate=False, dynamic=True)
        while not self.stopping_condition(gen, real_time_start):
            self.step(dynamic=self.dynamic)
            gen += 1
            if self.verbose and time.time() - display_timer > self.v_timer:
                self.step_info(gen, real_time_start)
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
        print(f"\tEvaluations of fitness: {self.objfunc.counter}")
        print(f"\tSubstrate probability:")
        subs_names = [i.evolution_method for i in self.substrates]
        weights = [round(i, 6) for i in self.population.substrate_weight]
        adjust = max([len(i) for i in subs_names])
        for idx, val in enumerate(subs_names):
            print(f"\t\t{val}:".ljust(adjust+3, " ") + f"{weights[idx]}")
        print()

    def display_report(self):
        factor = 1
        if self.opt == "min":
            factor = -1

        # Print Info
        print("Number of generations:", len(self.history))
        print("Real time spent: ", round(self.real_time_spent, 5), "s", sep="")
        print("CPU time spent: ", round(self.time_spent, 5), "s", sep="")
        print("Number of fitness evaluations:", self.objfunc.counter)
        print(f"\tSubstrate probability:")
        subs_names = [i.evolution_method for i in self.substrates]
        weights = [round(i, 6) for i in self.population.substrate_weight]
        adjust = max([len(i) for i in subs_names])
        for idx, val in enumerate(subs_names):
            print(f"\t\t{val}:".ljust(adjust+3, " ") + f"{weights[idx]}")
        
        best_fitness = factor * self.population.best_solution()[1]
        print("Best fitness:", best_fitness)

        # Plot fitness history
        #fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(10,5))
        #fig.suptitle("CRO_SL")

        plt.subplot(2, 2, 1)
        plt.plot(self.history, "blue")
        plt.xlabel("generations")
        plt.ylabel("fitness")
        plt.title("CRO_SL fitness")

        plt.subplot(2, 2, 2)
        m = np.array(self.population.substrate_history)[1:].T
        for i in m:
            plt.plot(factor * i)
        plt.legend([i.evolution_method for i in self.substrates])
        plt.xlabel("generations")
        plt.ylabel("fitness")
        plt.title("Fitness of each substrate")

        plt.subplot(2, 2, 3)
        prob_data = np.array(self.population.substrate_w_history).T
        plt.stackplot(range(prob_data.shape[1]), prob_data, labels=[i.evolution_method for i in self.substrates])
        plt.legend()
        plt.xlabel("generations")
        plt.ylabel("probability")
        plt.title("Probability of each substrate")
        plt.show()
        
def main():
    substrates_int = [SubstrateInt("SBX"), SubstrateInt("DGauss"), SubstrateInt("Perm"),
     SubstrateInt("1point"), SubstrateInt("2point"), SubstrateInt("Multipoint"), SubstrateInt("Xor")]
    #substrates_int = [SubstrateInt("DE/best/1"), SubstrateInt("DE/rand/1"), SubstrateInt("DE/rand/2"),
    # SubstrateInt("DE/best/2"), SubstrateInt("DE/current-to-best/1"), SubstrateInt("DE/current-to-rand/1"),
    # SubstrateInt("AddOne"), SubstrateInt("DGauss"), SubstrateInt("Perm"),
    # SubstrateInt("1point"), SubstrateInt("2point"), SubstrateInt("Multipoint"), SubstrateInt("Xor")]

    substrates_real = [
        SubstrateReal("SBX", {"F":0.5}),
        SubstrateReal("Perm", {"F":0.3}),
        SubstrateReal("1point"),
        SubstrateReal("2point"),
        SubstrateReal("Multipoint"),
        SubstrateReal("BLXalpha", {"F":0.8}),
        #SubstrateReal("Rand"),
        #SubstrateReal("DE/best/1", {"F":0.5, "Pr":0.8}),
        #SubstrateReal("DE/rand/1", {"F":0.5, "Pr":0.8}),
        #SubstrateReal("DE/best/2", {"F":0.5, "Pr":0.8}),
        #SubstrateReal("DE/rand/2", {"F":0.5, "Pr":0.8}),
        #SubstrateReal("DE/current-to-best/1", {"F":0.5, "Pr":0.8}),
        #SubstrateReal("DE/current-to-rand/1", {"F":0.5, "Pr":0.8}),
        #SubstrateReal("HS", {"F":0.5, "Pr":0.8}),
        SubstrateReal("Gauss", {"F":0.07})]
    
    params = {
        "ReefSize": 500,
        "rho": 0.6,
        "Fd": 0.3,
        "Pd": 0.4,
        "k": 5,
        "K": 20,

        "stop_cond": "ngen",
        "time_limit": 10.0,
        "Ngen": 200,
        "Neval": 4000,
        "fit_target": 1000,

        "verbose": True,
        "v_timer": 1,

        "dynamic": True
    }

    #objfunc = MaxOnes(1000)
    #objfunc = Test1(30)
    objfunc = Rosenbrock(1000)
    #objfunc = Rastrigin(30)

    #c = CRO_SL(DiophantineEq(100000, np.random.randint(-100, 100, size=100000), -12), substrates_int, params)
    #c = CRO_SL(MaxOnes(1000), substrates_int, params)
    #c = CRO_SL(MaxOnesReal(1000, "min"), substrates_real, params)
    #c = CRO_SL(objfunc, substrates_int, params)
    c = CRO_SL(objfunc, substrates_real, params)
    ind, fit = c.optimize()
    print(ind)
    c.display_report()
    

if __name__ == "__main__":
    main()

import numpy as np
from matplotlib import pyplot as plt
from .Population import *
import time

"""
Harmony Search algorithm optimization
"""
class HS:
    """
    Constructor of the Harmony Search algorithm
    """
    def __init__(self, objfunc, mutation_op, params):
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
        self.population = Population(objfunc, mutation_op, params)
        
        # Metrics
        self.history = []
        self.pop_size = []
        self.best_fitness = 0
        self.time_spent = 0
        self.real_time_spent = 0

    """
    One step of the algorithm
    """
    def step(self, progress, depredate=True, classic=False):
        self.population.evolve()

        self.population.selection()
        
        _, best_fitness = self.population.best_solution()
        self.history.append(best_fitness)
    
    """
    Stopping conditions given by a parameter
    """
    def stopping_condition(self, gen, time_start):
        stop = True
        if self.stop_cond == "neval":
            stop = self.objfunc.counter >= self.Neval
        elif self.stop_cond == "ngen":
            stop = gen >= self.Ngen
        elif self.stop_cond == "time":
            stop = time.time()-time_start >= self.time_limit
        elif self.stop_cond == "fit_target":
            if self.objfunc.opt == "max":
                stop = self.population.best_solution()[1] >= self.fit_target
            else:
                stop = self.population.best_solution()[1] <= self.fit_target

        return stop

    """
    Progress of the algorithm as a number between 0 and 1, 0 at the begining, 1 at the end
    """
    def progress(self, gen, time_start):
        prog = 0
        if self.stop_cond == "neval":
            prog = self.objfunc.counter/self.Neval
        elif self.stop_cond == "ngen":
            prog = gen/self.Ngen 
        elif self.stop_cond == "time":
            stop = (time.time()-time_start)/self.time_limit
        elif self.stop_cond == "fit_target":
            if self.objfunc.opt == "max":
                stop = self.population.best_solution()[1]/self.fit_target
            else:
                stop = self.fit_target/self.population.best_solution()[1]

        return prog

    """
    Execute the algorithm to get the best solution possible along with it's evaluation
    """
    def optimize(self):
        gen = 0
        time_start = time.process_time()
        real_time_start = time.time()
        display_timer = time.time()

        self.population.generate_random()
        while not self.stopping_condition(gen, real_time_start):
            prog = self.progress(gen, real_time_start)
            self.step(prog)
            gen += 1
            if self.verbose and time.time() - display_timer > self.v_timer:
                self.step_info(gen, real_time_start)
                display_timer = time.time()
                
        self.real_time_spent = time.time() - real_time_start
        self.time_spent = time.process_time() - time_start
        return self.population.best_solution()
    
    """
    Displays information about the current state of the algotithm
    """
    def step_info(self, gen, start_time):
        print(f"Time Spent {round(time.time() - start_time,2)}s:")
        print(f"\tGeneration: {gen}")
        best_fitness = self.population.best_solution()[1]
        print(f"\tBest fitness: {best_fitness}")
        print(f"\tEvaluations of fitness: {self.objfunc.counter}")
        print()
    
    """
    Shows a summary of the execution of the algorithm
    """
    def display_report(self, show_plots=True):
        # Print Info
        print("Number of generations:", len(self.history))
        print("Real time spent: ", round(self.real_time_spent, 5), "s", sep="")
        print("CPU time spent: ", round(self.time_spent, 5), "s", sep="")
        print("Number of fitness evaluations:", self.objfunc.counter)
        
        best_fitness = self.population.best_solution()[1]
        print("Best fitness:", best_fitness)

        if show_plots:
            # Plot fitness history            
            plt.plot(self.history, "blue")
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("Genetic fitness")
            plt.show()
import time
import os
import warnings

import numpy as np
from PyCROSL.CoralPopulation import CoralPopulation
from matplotlib import pyplot as plt
# import json


class CRO_SL:
    """
    Coral reef optimization with substrate layers

    Parameters:
        popSize: maximum number of corals in the reef        [Integer,          100  recomended]
        rho: percentage of initial ocupation of the reef     [Real from 0 to 1, 0.6  recomended]
        Fb: broadcast spawning proportion                    [Real from 0 to 1, 0.98 recomended]
        Fd: depredation proportion                           [Real from 0 to 1, 0.2  recomended]
        Pd: depredation probability                          [Real from 0 to 1, 0.9  recomended]
        k: maximum attempts for larva setting                [Integer,          4    recomended]
        K: maximum amount of corals with duplicate solutions [Integer,          20   recomended]

        group_subs: evolve only within the same substrate or with the whole population

        dynamic: change the size of each substrate
        dyn_method: which value to use for the evaluation of the substrates
        dyn_metric: how to process the data from the substrate for the evaluation
        dyn_steps: number of times the substrates will be evaluated
        prob_amp: parameter that makes probabilities more or less extreme with the same data

    """

    def __init__(self, objfunc, substrates, params):
        self.params = params

        # Dynamic parameters
        self.dynamic = params["dynamic"]
        self.dyn_method = params["dyn_method"]

        # Verbose parameters
        self.verbose = params["verbose"]
        self.v_timer = params["v_timer"]

        # Parallelization parameters
        self.Njobs = params["Njobs"] if "Njobs" in params else 1

        # Stopping conditions
        self.stop_cond = params["stop_cond"]
        self.Ngen = params["Ngen"]
        self.Neval = params["Neval"]
        self.time_limit = params["time_limit"]
        self.fit_target = params["fit_target"]

        # Data structures of the algorithm
        self.objfunc = objfunc
        self.substrates = substrates
        self.population = CoralPopulation(objfunc, substrates, params)
        
        # Metrics
        self.history = []
        self.sol_history = []
        self.pop_size = []
        self.best_fitness = 0
        self.time_spent = 0
        self.real_time_spent = 0

    
    def restart(self):
        """
        Resets the data of the CRO algorithm
        """
        
        self.population = CoralPopulation(self.objfunc, self.substrates, self.params)
        self.history = []
        self.pop_size = []
        self.best_fitness = 0
        self.time_spent = 0
        self.real_time_spent = 0
        self.objfunc.counter = 0

    
    def step(self, progress, depredate=True, classic=False):        
        """
        One step of the algorithm
        """
        
        if not classic:
            self.population.generate_substrates(progress)

        larvae = self.population.evolve_with_substrates()

        larvae = self.population.evaluate_fitnesses(larvae, self.Njobs)
        
        self.population.larvae_setting(larvae)

        if depredate:
            self.population.extreme_depredation()
            self.population.depredation()
        
        best_indiv, best_fitness = self.population.best_solution()
        self.history.append(best_fitness)
        self.sol_history.append(best_indiv)
    
    
    def local_search(self, operator, n_ind, iterations=100):
        """
        Local search
        """
        
        if self.verbose:
            print(f"Starting local search, {n_ind} individuals searching {iterations} neighbours each.")
        
        self.population.local_search(operator, n_ind, iterations)
        return self.population.best_solution()

    
    def stopping_condition(self, gen, time_start):
        """
        Stopping conditions given by a parameter
        """
        
        stop = True
        if self.stop_cond == "Neval":
            stop = self.objfunc.counter >= self.Neval
        elif self.stop_cond == "Ngen":
            stop = gen >= self.Ngen
        elif self.stop_cond == "time":
            stop = time.time()-time_start >= self.time_limit
        elif self.stop_cond == "fit_target":
            if self.objfunc.opt == "max":
                stop = self.population.best_solution()[1] >= self.fit_target
            else:
                stop = self.population.best_solution()[1] <= self.fit_target

        return stop

    
    def progress(self, gen, time_start):
        """
        Progress of the algorithm as a number between 0 and 1, 0 at the begining, 1 at the end
        """
        
        prog = 0
        if self.stop_cond == "Neval":
            prog = self.objfunc.counter/self.Neval
        elif self.stop_cond == "Ngen":
            prog = gen/self.Ngen 
        elif self.stop_cond == "time":
            prog = (time.time()-time_start)/self.time_limit
        elif self.stop_cond == "fit_target":
            if self.objfunc.opt == "max":
                prog = self.population.best_solution()[1]/self.fit_target
            else:
                prog = self.fit_target/self.population.best_solution()[1]

        return prog
    
    
    def save_data(self, solution_file="best_solution.csv", population_file="last_population.csv", history_file="fit_history.csv", prob_file="prob_history.csv", indiv_history="indiv_history.csv"):
        """
        Save execution data
        """
        
        ind, fit = self.population.best_solution()
        np.savetxt(solution_file, ind.reshape([1, -1]), delimiter=',')
        
        pop_matrix = [i.solution for i in self.population.population]
        np.savetxt(population_file, pop_matrix, delimiter=',')

        np.savetxt(history_file, self.history)

        np.savetxt(indiv_history, self.sol_history)

        if self.dynamic:
            prob_data = np.array(self.population.substrate_w_history)
            np.savetxt(prob_file, prob_data, delimiter=',')

        # with open(file_name, "a") as file:
        #     file.write(str(fit))
        


    
    def optimize(self, save_data = True, update_data = True):
        """
        Execute the algorithm to get the best solution possible along with it's evaluation
        """
        
        gen = 0
        time_start = time.process_time()
        real_time_start = time.time()
        display_timer = time.time()

        self.population.generate_random()
        self.step(0, depredate=False)
        while not self.stopping_condition(gen, real_time_start):
            prog = self.progress(gen, real_time_start)
            self.step(prog, depredate=True)
            gen += 1
            if self.verbose and time.time() - display_timer > self.v_timer:
                self.step_info(gen, real_time_start)
                display_timer = time.time()
            
            if update_data:
                self.save_data()
                
        self.real_time_spent = time.time() - real_time_start
        self.time_spent = time.process_time() - time_start
        if save_data:
            self.save_data()
        return self.population.best_solution()
    
    
    def safe_optimize(self):
        """
        Execute the algorithm with early stopping
        """
        
        result = (np.array([]), 0)
        try:
            result = self.optimize()
        except KeyboardInterrupt:
            print("stopped early")
            self.save_solution(file_name="stopped.csv")
            self.display_report(show_plots=False, save_figure=True, figure_name="stopped.eps")
            exit(1)
        
        return result

    
    def optimize_classic(self, save_data = True, update_data = True):
        """
        Execute the classic version of the algorithm
        """
        
        gen = 0
        time_start = time.process_time()
        real_time_start = time.time()
        display_timer = time.time()

        self.population.generate_random()
        self.step(0, depredate=False)
        while not self.stopping_condition(gen, real_time_start):
            prog = self.progress(gen, real_time_start)
            self.step(prog, depredate=True, classic=True)
            gen += 1
            if self.verbose and time.time() - display_timer > self.v_timer:
                self.step_info(gen, real_time_start)
                display_timer = time.time()
            
            if update_data:
                self.save_data()
                
        self.real_time_spent = time.time() - real_time_start
        self.time_spent = time.process_time() - time_start
        if save_data:
            self.save_data()
        return self.population.best_solution()

    
    def best_solution(self):
        """
        Gets the best solution so far in the population
        """
        
        return self.population.best_solution()

    
    def save_solution(self, file_name="solution.csv"):
        """
        Save the result of an execution to a csv file in disk
        """
        
        ind, fit = self.population.best_solution()
        np.savetxt(file_name, ind.reshape([1, -1]), delimiter=',')
        with open(file_name, "a") as file:
            file.write(str(fit))

    
    def step_info(self, gen, start_time):
        """
        Displays information about the current state of the algotithm
        """
        
        print(f"Time Spent {round(time.time() - start_time,2)}s:")
        print(f"\tGeneration: {gen}")
        best_fitness = self.population.best_solution()[1]
        print(f"\tBest fitness: {best_fitness}")
        print(f"\tEvaluations of fitness: {self.objfunc.counter}")

        if self.dynamic:
            print(f"\tSubstrate probability:")
            subs_names = [i.evolution_method for i in self.substrates]
            weights = [round(i, 6) for i in self.population.substrate_weight]
            adjust = max([len(i) for i in subs_names])
            for idx, val in enumerate(subs_names):
                print(f"\t\t{val}:".ljust(adjust+3, " ") + f"{weights[idx]}")
        print()
    
    
    def display_report(self, show_plots=True, save_figure=False, figure_name="fig.eps"):
        """
        Shows a summary of the execution of the algorithm
        """
        
        if save_figure:
            if not os.path.exists("figures"):
                os.makedirs("figures")

        if self.dynamic:
            self.display_report_dyn(show_plots, save_figure, figure_name)
        else:
            self.display_report_nodyn(show_plots, save_figure, figure_name)

    
    def display_report_dyn(self, show_plots=True, save_figure=False, figure_name="fig.eps"):
        """
        Version of the summary for the dynamic variant
        """
        
        factor = 1
        if self.objfunc.opt == "min" and self.dyn_method != "success":
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
        
        best_fitness = self.population.best_solution()[1]
        print("Best fitness:", best_fitness)

        
        if save_figure:
            # Plot fitness history            
            plt.plot(self.history, "blue")
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("DPCRO_SL fitness")
            plt.savefig(f"figures/fit_{figure_name}")
            plt.cla()
            plt.clf()
            
            m = np.array(self.population.substrate_history)[1:].T
            for i in m:
                plt.plot(factor * i)
            plt.legend([i.evolution_method for i in self.substrates])
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("Evaluation of each substrate")
            plt.savefig(f"figures/eval_{figure_name}")
            plt.cla()
            plt.clf()
            

            prob_data = np.array(self.population.substrate_w_history).T
            plt.stackplot(range(prob_data.shape[1]), prob_data, labels=[i.evolution_method for i in self.substrates])
            plt.legend([i.evolution_method for i in self.substrates])
            plt.xlabel("generations")
            plt.ylabel("probability")
            plt.title("Probability of each substrate")
            plt.savefig(f"figures/prob_{figure_name}")
            plt.cla()
            plt.clf()


            plt.plot(prob_data.T)
            plt.legend([i.evolution_method for i in self.substrates])
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("Evaluation of each substrate")
            plt.savefig(f"figures/subs_{figure_name}")
            plt.cla()
            plt.clf()

            plt.close("all")
            
        if show_plots:
            # Plot fitness history            
            fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(10,10))
            fig.suptitle("DPCRO_SL")
            plt.subplot(2, 2, 1)
            
            plt.plot(self.history, "blue")
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("DPCRO_SL fitness")

            
            plt.subplot(2, 2, 2)
            m = np.array(self.population.substrate_history)[1:].T
            for i in m:
                plt.plot(factor * i)
            plt.legend([i.evolution_method for i in self.substrates])
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("Evaluation of each substrate")

            plt.subplot(2, 1, 2)
            prob_data = np.array(self.population.substrate_w_history).T
            plt.stackplot(range(prob_data.shape[1]), prob_data, labels=[i.evolution_method for i in self.substrates])
            plt.legend([i.evolution_method for i in self.substrates])
            plt.xlabel("generations")
            plt.ylabel("probability")
            plt.title("Probability of each substrate")

            plt.show()

    
    def display_report_nodyn(self, show_plots=True, save_figure=False, figure_name="fig.eps"):
        """
        Version of the summary for the dynamic variant
        """
        
        factor = 1
        if self.objfunc.opt == "min":
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
        
        best_fitness = self.population.best_solution()[1]
        print("Best fitness:", best_fitness)

        if save_figure:            
            plt.plot(self.history, "blue")
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("PCRO_SL fitness")
            plt.savefig(f"figures/fit_{figure_name}")
            plt.cla()
            plt.clf()

            
            m = np.array(self.population.substrate_history)[1:].T
            for i in m:
                plt.plot(factor * i)
            plt.legend([i.evolution_method for i in self.substrates])
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("Evaluation of each substrate")
            plt.savefig(f"figures/eval_{figure_name}")
            plt.cla()
            plt.clf()
            plt.close("all")

        if show_plots:
            # Plot fitness history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
            fig.suptitle("CRO_SL")
            plt.subplot(1, 2, 1)
            
            plt.plot(self.history, "blue")
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("PCRO_SL fitness")

            plt.subplot(1, 2, 2)
            m = np.array(self.population.substrate_history)[1:].T
            for i in m:
                plt.plot(factor * i)
            plt.legend([i.evolution_method for i in self.substrates])
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("Evaluation of each substrate")

            if save_figure:
                plt.savefig(f"figures/{figure_name}")

            plt.show()

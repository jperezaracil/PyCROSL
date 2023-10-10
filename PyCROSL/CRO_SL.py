import time
import os
import warnings

import numpy as np
from PyCROSL.CoralPopulation import CoralPopulation
from matplotlib import pyplot as plt
import pyparsing as pp
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

    def __init__(self, objfunc, substrates, params, file_name_extra=""):
        self.params = params

        # Dynamic parameters
        self.dynamic = params.get("dynamic", True) 
        self.dyn_method = params.get("dyn_method", "fitness")

        # Verbose parameters
        self.verbose = params.get("verbose", True) 
        self.v_timer = params.get("v_timer", 1) 
        self.file_name_extra = file_name_extra

        # Parallelization parameters
        self.Njobs = params.get("Njobs", 1) 

        # Stopping conditions
        self.stop_cond = params.get("stop_cond", "time_limit")
        self.stop_cond_parsed = parse_stopping_cond(self.stop_cond)

        self.Ngen = params.get("Ngen", 100) 
        self.Neval = params.get("Neval", 1e5) 
        self.time_limit = params.get("time_limit", 100) 
        self.fit_target = params.get("fit_target", 0) 

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

        neval_reached = self.objfunc.counter >= self.Neval

        ngen_reached = gen >= self.Ngen

        real_time_reached = time.time() - time_start >= self.time_limit

        if self.objfunc.opt == "max":
            target_reached = self.best_solution()[1] >= self.fit_target
        else:
            target_reached = self.best_solution()[1] <= self.fit_target

        return process_condition(self.stop_cond_parsed, neval_reached, ngen_reached, real_time_reached, target_reached)

    
    def progress(self, gen, time_start):
        """
        Progress of the algorithm as a number between 0 and 1, 0 at the begining, 1 at the end
        """

        neval_reached = self.objfunc.counter/self.Neval
        
        ngen_reached = gen/self.Ngen

        real_time_reached = (time.time() - time_start)/self.time_limit

        best_fitness = self.best_solution()[1]
        if self.objfunc.opt == "max":
            target_reached = best_fitness/self.fit_target
        else:
            if best_fitness == 0:
                best_fitness = 1e-15
            target_reached = self.fit_target/best_fitness

        return process_progress(self.stop_cond_parsed, neval_reached, ngen_reached, real_time_reached, target_reached)
    
    
    def save_data(self, solution_file=None, population_file=None, history_file=None, prob_file=None, indiv_history=None):
        """
        Save execution data
        """

        postfix = "" if self.file_name_extra == "" else "_" + self.file_name_extra

        if solution_file is None:
            solution_file = f"best_solution{postfix}.csv"

        if population_file is None:
            population_file = f"last_population{postfix}.csv"

        if history_file is None:
            history_file = f"fit_history{postfix}.csv"
        
        if prob_file is None:
            prob_file = f"prob_history{postfix}.csv"

        if indiv_history is None:
            indiv_history = f"indiv_history{postfix}.csv"
        
        
        ind, fit = self.population.best_solution()
        np.savetxt(solution_file, ind.reshape([1, -1]), delimiter=',')
        
        pop_matrix = [i.solution for i in self.population.population]
        np.savetxt(population_file, pop_matrix, delimiter=',')

        np.savetxt(history_file, self.history)

        np.savetxt(indiv_history, self.sol_history)

        if self.dynamic:
            prob_data = np.array(self.population.substrate_w_history)
            np.savetxt(prob_file, prob_data, delimiter=',')


    def optimize(self, save_data = True, update_data = True):
        """
        Execute the algorithm to get the best solution possible along with it's evaluation
        """

        if self.verbose:
            self.init_info()

        gen = 0
        time_start = time.process_time()
        real_time_start = time.time()
        display_timer = time.time()

        self.population.generate_random()

        if self.verbose:
            self.population.evaluate_fitnesses(self.population.population, self.Njobs)
            self.step_info(gen, real_time_start)

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
            postfix = "" if self.file_name_extra == "" else "_" + self.file_name_extra
            self.save_solution(file_name=f"stopped{postfix}.csv")
            self.display_report(show_plots=False, save_figure=True, figure_name=f"stopped{postfix}.eps")
            exit(1)

        return result

    
    def optimize_classic(self, save_data = True, update_data = True):
        """
        Execute the classic version of the algorithm
        """

        if self.verbose:
            self.init_info()

        gen = 0
        time_start = time.process_time()
        real_time_start = time.time()
        display_timer = time.time()

        self.population.generate_random()

        if self.verbose:
            self.population.evaluate_fitnesses(self.population.population, self.Njobs)
            self.step_info(gen, real_time_start)

        self.step(0, depredate=False)

        if self.verbose:
            self.step_info(gen, real_time_start)

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

    
    def init_info(self):
        print(f"Starting optimization of {self.objfunc.name}")
        print(f"-------------------------{'-'*len(self.objfunc.name)}")
        print()

    
    def step_info(self, gen, start_time):
        """
        Displays information about the current state of the algotithm
        """

        print(f"Optimizing {self.objfunc.name}:")
        print(f"\tTime Spent: {round(time.time() - start_time,2)}s")
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
        print(f"Finished optimizing {self.objfunc.name}")
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
        print(f"Finished optimizing {self.objfunc.name}")
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


def parse_stopping_cond(condition_str):
    """
    This function parses an expression of the form "neval or cpu_time" into
    a tree structure so that it can be futher processed.
    """

    orop = pp.Literal("and")
    andop = pp.Literal("or")
    condition = pp.oneOf(["Neval", "Ngen", "time_limit", "fit_target"])

    expr = pp.infixNotation(
        condition,
        [
            (orop, 2, pp.opAssoc.RIGHT),
            (andop, 2, pp.opAssoc.RIGHT)
        ]
    )

    return expr.parse_string(condition_str).as_list()


def process_condition(cond_parsed, neval, ngen, real_time, target):
    """
    This function receives as an input an expression for the stopping condition
    and the truth variable of the possible stopping conditions and returns wether to stop or not.
    """

    result = None

    if isinstance(cond_parsed, list):
        if len(cond_parsed) == 3:
            cond1 = process_condition(cond_parsed[0], neval, ngen, real_time, target)
            cond2 = process_condition(cond_parsed[2], neval, ngen, real_time, target)

            if cond_parsed[1] == "or":
                result = cond1 or cond2
            elif cond_parsed[1] == "and":
                result = cond1 and cond2

        elif len(cond_parsed) == 1:
            result = process_condition(cond_parsed[0], neval, ngen, real_time, target)

    else:
        if cond_parsed == "Neval":
            result = neval
        elif cond_parsed == "Ngen":
            result = ngen
        elif cond_parsed == "time_limit":
            result = real_time
        elif cond_parsed == "fit_target":
            result = target
    
    return result


def process_progress(cond_parsed, neval, ngen, real_time, target):
    """
    This function receives as an input an expression for the stopping condition 
    and the truth variable of the possible stopping conditions and returns wether to stop or not. 
    """
    result = None
    
    if isinstance(cond_parsed, list):
        if len(cond_parsed) == 3:
            
            progress1 = process_progress(cond_parsed[0], neval, ngen, real_time, target)
            progress2 = process_progress(cond_parsed[2], neval, ngen, real_time, target)

            if cond_parsed[1] == "or":
                result = max(progress1, progress2)
            elif cond_parsed[1] == "and":
                result = min(progress1, progress2)
            
        elif len(cond_parsed) == 1:
            result = process_progress(cond_parsed[0], neval, ngen, real_time, target)
    else:
        if cond_parsed == "Neval":
            result = neval
        elif cond_parsed == "Ngen":
            result = ngen
        elif cond_parsed == "time_limit":
            result = real_time
        elif cond_parsed == "fit_target":
            result = target

    return result
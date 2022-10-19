import numpy as np
import random
from Substrate import *
from operators import *


"""
Substrate class that has continuous mutation and cross methods
"""
class SubstrateReal(Substrate):
    def __init__(self, evolution_method, params = None):
        self.evolution_method = evolution_method
        super().__init__(self.evolution_method, params)
    
    """
    Evolves a solution with a different strategy depending on the type of operator
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
        elif self.evolution_method == "Multicross":
            result = multiCross(solution.solution.copy(), others, self.params["n_ind"])
        elif self.evolution_method == "Gauss":
            result = gaussian(solution.solution.copy(), self.params["F"])
        elif self.evolution_method == "Laplace":
            result = laplace(solution.solution.copy(), self.params["F"])
        elif self.evolution_method == "Cauchy":
            result = cauchy(solution.solution.copy(), self.params["F"])
        elif self.evolution_method == "BLXalpha":
            result = blxalpha(solution.solution.copy(), solution2.solution.copy(), self.params["F"])
        elif self.evolution_method == "SBX":
            result = sbx(solution.solution.copy(), solution2.solution.copy(), self.params["F"])
        elif self.evolution_method == "Perm":
            result = permutation(solution.solution.copy(), self.params["F"])
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
        elif self.evolution_method == "DE/current-to-pbest/1":
            result = DECurrentToPBest1(solution.solution.copy(), others, self.params["F"], self.params["Pr"])
        elif self.evolution_method == "LSHADE":
            self.params["Pr"] = np.random.normal(self.params["Pr"], 0.1)
            self.params["F"] = np.random.normal(self.params["F"], 0.1)

            self.params["Pr"] = np.clip(self.params["Pr"], 0, 1)
            self.params["F"] = np.clip(self.params["F"], 0, 1)

            result = DECurrentToPBest1(solution.solution.copy(), others, self.params["F"], self.params["Pr"])
            #result = crossMp(result, solution2.solution.copy())
            #result = blxalpha(result, solution2.solution.copy(), self.params["Pr"])
            
        elif self.evolution_method == "SA":
            result = sim_annealing(solution, self.params["F"], objfunc, self.params["temp_ch"], self.params["iter"])
        elif self.evolution_method == "HS":
            result = harmony_search(solution.solution.copy(), population, self.params["F"], self.params["Pr"], 0.4)
        elif self.evolution_method == "Replace":
            result = replace(solution.solution.copy(), population, self.params["method"], self.params["F"])
        elif self.evolution_method == "Dummy":
            result = dummy_op(solution.solution.copy(), self.params["F"])
        else:
            print(f"Error: evolution method \"{self.evolution_method}\" not defined")
            exit(1)
            
        
        return result

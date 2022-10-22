import random

import numpy as np
from operators import *
from Substrate import *

"""
Substrate class that has discrete mutation and cross methods
"""
class SubstrateInt(Substrate):
    def __init__(self, evolution_method, params = None):
        self.evolution_method = evolution_method
        super().__init__(self.evolution_method, params)
    
    """
    Applies a mutation method depending on the type of operator
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
        elif self.evolution_method == "Perm":
            result = permutation(solution.solution.copy(), self.params["F"])
        elif self.evolution_method == "Xor":
            result = xorMask(solution.solution.copy(), self.params["F"])
        elif self.evolution_method == "Gauss":
            result = gaussian(solution.solution.copy(), self.params["F"])
        elif self.evolution_method == "Laplace":
            result = laplace(solution.solution.copy(), self.params["F"])
        elif self.evolution_method == "Cauchy":
            result = cauchy(solution.solution.copy(), self.params["F"])
        elif self.evolution_method == "AddOne":
            result = addOne(solution.solution.copy(), self.params["F"])
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
        elif self.evolution_method == "SA":
            result = sim_annealing(solution, self.params["F"], objfunc, self.params["temp_ch"], self.params["iter"])
        elif self.evolution_method == "HS":
            result = harmony_search(solution.solution.copy(), population, self.params["F"], self.params["Pr"], 0.4)
        elif self.evolution_method == "Replace":
            result = random_replace(solution.solution.copy())
        else:
            print(f"Error: evolution method \"{self.evolution_method}\" not defined")
            exit(1)

        return np.round(result).astype(np.int32)
import numpy as np
import random

"""
Abstract Operator class
"""
class Operator:
    def __init__(self, evolution_method, params):
        self.evolution_method = evolution_method
        self.params = params
        if self.params is None:
            self.params = {"F": 0.5, "Pr": 0.8, "temp_ch":10, "iter":20}
    
    """
    Evolves a solution with a different strategy depending on the type of operator
    """
    def evolve(self, solution, strength, population, objfunc):
        pass

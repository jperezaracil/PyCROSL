import random
import numpy as np
from abc import ABC, abstractmethod



class Substrate(ABC):
    """
    Abstract Substrate class
    """

    def __init__(self, evolution_method, params):
        self.evolution_method = evolution_method
        self.params = params
        if self.params is None:
            # Default parameters
            self.params = {
                "F": 0.5, 
                "Cr": 0.8,
                "Par":0.1,
                "N":5,
                "method": "Gauss",
                "temp_ch":10,
                "iter":20
            }
    
    @abstractmethod
    def evolve(self, solution, strength, population, objfunc):
        """
        Evolves a solution with a different strategy depending on the type of substrate
        """

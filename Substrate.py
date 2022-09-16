import numpy as np
import random

"""
Abstract Substrate class
"""
class Substrate:
    def __init__(self, evolution_method):
        self.evolution_method = evolution_method
    
    """
    Evolves a solution with a different strategy depending on the type of substrate
    """
    def evolve(self, solution, strength, population):
        pass

import numpy as np
from AbsObjetiveFunc import AbsObjetiveFunc
from numba import jit

"""
Example of objective function.
"""

class Sphere(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return sphere(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def check_bounds(self, solution):
        return np.clip(solution, -100, 100)

class HighCondElliptic(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return high_cond_elipt_f(solution)
    
    def random_solution(self):
        return 10.24*np.random.random(self.size)-5.12
    
    def check_bounds(self, solution):
        return np.clip(solution, -5.12, 5.12)

class BentCigar(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return bent_cigar(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def check_bounds(self, solution):
        return np.clip(solution, -100, 100)

class Discus(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return discus(solution)
    
    def random_solution(self):
        return 10.24*np.random.random(self.size)-5.12
    
    def check_bounds(self, solution):
        return np.clip(solution, -5.12, 5.12)

class Rosenbrock(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return rosenbrock(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def check_bounds(self, solution):
        return np.clip(solution, -100, 100)

class Ackley(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return ackley(solution)
    
    def random_solution(self):
        return 10.24*np.random.random(self.size)-5.12
    
    def check_bounds(self, solution):
        return np.clip(solution, -5.12, 5.12)

class Weierstrass(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return weierstrass(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def check_bounds(self, solution):
        return np.clip(solution, -100, 100)

class Griewank(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return griewank(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def check_bounds(self, solution):
        return np.clip(solution, -100, 100)

class Rastrigin(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return rastrigin(solution)
    
    def random_solution(self):
        return 10.24*np.random.random(self.size)-5.12
    
    def check_bounds(self, solution):
        return np.clip(solution, -5.12, 5.12)

class ModSchwefel(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return mod_schwefel(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def check_bounds(self, solution):
        return np.clip(solution, -100, 100)

class Katsuura(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return katsuura(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def check_bounds(self, solution):
        return np.clip(solution, -100, 100)

class HappyCat(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return happy_cat(solution)
    
    def random_solution(self):
        return 4*np.random.random(self.size)-2
    
    def check_bounds(self, solution):
        return np.clip(solution, -2, 2)

class HGBat(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return hgbat(solution)
    
    def random_solution(self):
        return 4*np.random.random(self.size)-2
    
    def check_bounds(self, solution):
        return np.clip(solution, -2, 2)

class ExpandedGriewankPlusRosenbrock(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return exp_griewank_plus_rosenbrock(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def check_bounds(self, solution):
        return np.clip(solution, -100, 100)

class ExpandedShafferF6(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return exp_shafferF6(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def check_bounds(self, solution):
        return np.clip(solution, -100, 100)


@jit(nopython=True)
def sphere(solution):
    return (solution**2).sum()

#@jit(nopython=True)
def high_cond_elipt_f(vect):
    d = vect.shape[0]
    if d > 1:
        c = np.array([1e6**((i-1)/(d-1)) for i in range(1,d+1)])
    else:
        c = 1e6
    return (c*vect*vect).sum()

@jit(nopython=True)
def bent_cigar(solution):
    return solution[0]**2 + 1e6*(solution[1:]**2).sum()

@jit(nopython=True)
def discus(solution):
    return 1e6*solution[0]**2 + (solution[1:]**2).sum()

@jit(nopython=True)
def rosenbrock(solution):
    term1 = solution[1:] - solution[:-1]**2
    term2 = 1 - solution[:-1]
    result = 100*term1**2 + term2**2
    return result.sum()

@jit(nopython=True)
def ackley(solution):
    term1 = (solution**2).sum()
    term1 = -0.2 * np.sqrt(term1/solution.size)
    term2 = (np.cos(2*np.pi*solution)).sum()/solution.size
    return np.exp(1) - 20 * np.exp(term1) - np.exp(term2) + 20

#@jit(nopython=False)
def weierstrass(solution, iter=20):
    return np.sum(np.array([0.5**k * np.cos(2*np.pi*3**k*(solution+0.5)) for k in range(iter)]))

#@jit(nopython=True)
def griewank(solution):
    term1 = (solution**2).sum()
    term2 = np.prod(np.cos(solution/np.sqrt(np.arange(1, solution.size+1))))
    return 1 + term1/4000 - term2

@jit(nopython=True)
def rastrigin(solution, A=10):
    return (A * len(solution) + (solution**2 - A*np.cos(2*np.pi*solution)).sum())

@jit(nopython=True)
def mod_schwefel(solution):
    fit = 0
    for i in range(solution.size):
        z = solution[i] + 4.209687462275036e2
        if z > 500:
            fit = fit - (500 - z % 500) * np.sin((500 - z%500)** 0.5)
            tmp = (z - 500) / 100
            fit = fit + tmp * tmp / solution.size
        elif z < -500:
            fit = fit - (-500 - abs(z) % 500) * np.sin((500 - abs(z)%500)** 0.5)
            tmp = (z + 500) / 100
            fit = fit + tmp * tmp / solution.size
        else:
            fit = fit - z * np.sin(abs( z )**0.5)
    return fit + 4.189828872724338e2 * solution.size

#@jit(nopython=True)
def katsuura(solution):
    return 10/solution.size**2 * np.prod([1 + (i+1)*np.sum((np.abs(2**(np.arange(1, 32+1))*solution[i]-np.round(2**(np.arange(1, 32+1))*solution[i]))*2**(-np.arange(1, 32+1, dtype=float)))**(10/solution.size**1.2)) for i in range(solution.size)]) - 10/solution.size**2

@jit(nopython=True)
def happy_cat(solution):
    z = solution+4.189828872724338e2
    r2 = (z * solution).sum()
    s = solution.sum()
    return np.abs(r2-solution.size)**0.25 + (0.5*r2+s)/solution.size + 0.5
    #return ((solution**2).sum() - solution.size)**0.25 + (0.5*(solution**2).sum() + solution.sum())/solution.size + 0.5

#@jit(nopython=True)
def hgbat(solution):
    z = solution+4.189828872724338e2
    r2 = (z * solution).sum()
    s = solution.sum()
    return np.abs((r2**2-s**2))**0.5 + (0.5*r2 + s)/solution.size + 0.5

#@jit(nopython=True)
def exp_griewank_plus_rosenbrock(solution):
    z = solution[:-1]+4.189828872724338e2
    tmp1 = solution[:-1]**2-solution[1:]
    tmp2 = z - 1
    tmp = 100*tmp1**2 + tmp2**2
    grw = (tmp**2/4000 - np.cos(tmp) + 1).sum()

    term1 = solution[1:] - solution[:-1]**2
    term2 = 1 - solution[:-1]
    ros = (100*term1**2 + term2**2).sum()
    
    return grw + ros**2/4000 - np.cos(ros) + 1

#@jit(nopython=True)
def exp_shafferF6(solution):
    term1 = np.sin(np.sqrt((solution[:-1]**2 + solution[1:]**2).sum()))**2 - 0.5
    term2 = 1 + 0.001*(solution[:-1]**2 + solution[1:]**2).sum()
    temp = 0.5 + term1/term2

    term1 = np.sin(np.sqrt(((solution.size - 1)**2 + solution[0]**2).sum()))**2 - 0.5
    term2 = 1 + 0.001*((solution.size - 1)**2 + solution[0]**2)

    return temp + 0.5 + term1/term2

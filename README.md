# CRO-SL
Python implementation of the Coral reef optimization algorithm with substrate layers

To configure the hyperparameters a dictionaty will have to be given to the class CRO_SL.

The hyperparameters will be the following:
- Algorithm's hyperparameters:
    - popSize: maximum number of corals in the reef       
    - rho: percentage of initial ocupation of the reef 
    - Fb: broadcast spawning proportion     
    - Fd: depredation proportion          
    - Pd: depredation probability     
    - k: maximum attempts for larva setting               
    - K: maximum amount of corals with duplicate solutions
    - group_subs: evolve only within the same substrate or with the whole population
- Dynamic variant hyperparameters:
    - dynamic: use or not the dynamic variant of the algotithm
    - method: value that is used to determine the probability of choosing each substrate
        - "fitness": uses the fitness of the individuals of each substrate.
        - "diff": uses the difference between the fitness of the previous generation and the current one.
        - "success": uses the ratio of successful larvae in each generation.
    - dyn_metric: how to agregate the values of each substrate to get the metric of each of them
        - "best": takes the best fitness
        - "avg": takes the average fitness
        - "med": takes the median fitness
        - "worse": takes the worse fitness
    - dyn_steps: specifies the number of times the substrates will be evaluated, -1 for every generation
    - prob_amp: how differences between de metric affect the probability of each one, lower means more amplification
- Stopping conditions:
    - stop_cond: stopping condition, there are various options
        - "neval": stop after a given number of evaluations of the fitness function
        - "ngen": stop after a given number of generations
        - "time": stop after a fixed amount of execution time (real time, not CPU time)
        - "fit_target": stop after reaching a desired fitness, accounting for whether we have maximization and minimization
    - Neval: number of evaluations of the fitness function
    - Ngen: number of generations
    - time_limit: execution time limit given in seconds
    - fit_target: value of the fitness function we want to reach
- Display options:
    - verbose: shows a report of the state of the algorithm periodicaly
    - v_timer: amount of time between each report

# PyCRO-SL

## About

This project is a Python implementation of the **Coral Reef Optimization with Substrate Layers** algorithm, 
both in its original version (CRO-SL) and in its probabilistic versions (PCRO-SL and DPCRO-SL). The goal is to provide an
easy-to-use, versatile Python package that can be used **off-the-shelf** for a wide array of optimization problems. 
Among its features, we highlight:

- *Operators included*: the package includes more than 30 solution operators, such as Gaussian noise addition, BLX Alpha, Multi-point crossover, etc.
- *Multiple conditions*: the algorithm can be configured to stop after a given number of evaluations, generations, or time, or after reaching a given fitness value. These stopping conditions can also be combined using any logical expression.
- *Parallelization*: the package allows for parallelization of the fitness function evaluations, which can significantly speed up optimization.
- *Automatic report*: the implementation produces a graphical report at the end of each simulation, summarizing the behavior of the most important metrics.  
- *Dynamic variant*: the algorithm has a dynamic variant, which allows it to reward the operators that perform better in the current problem.

For an explanation of how the CRO-SL algorithm works, we refer to [the paper](https://www.mdpi.com/2227-7390/11/7/1666).

## How to install

We recommend installing the project as a package. To do so, use the following commands:

```
git clone https://github.com.jperezaracil/PyCROSL.git
cd PyCROSL
pip intall -r requirements.txt
pip install -e .
```

Then, you can import the algorithm as a package, for example

```
from PyCROSL.CRO_SL import CRO_SL
```
## How to use it

To use PyCRO-SL in your own optimization problems, take a look at our [tutorial](/Tutorials/guide.ipynb).

## Parameters

To configure the hyperparameters a dictionary will have to be given to the class CRO_SL.
This dictionary contains the following parameters:

- Basic hyperparameters:
    - `popSize`: maximum number of corals in the reef       
    - `rho`: percentage of initial occupation of the reef 
    - `Fb`: broadcast spawning proportion
    - `Fd`: depredation proportion
    - `Pd`: depredation probability
    - `k`: maximum attempts for larva setting               
    - `K`: maximum number of corals with duplicate solutions
    - `group_subs`: if `True`, corals reproduce only within the same substrate, if `False` they reproduce within the whole population
- Dynamic variant hyperparameters:
    - `dynamic`: boolean value that determines whether to use the dynamic variant of the algorithm
    - `method`: string that determines how to determine the probability of choosing each substrate. Possible values:
        - `"fitness"`: uses the fitness of the individuals of each substrate.
        - `"diff"`: uses the difference between the fitness of the previous generation and the current one.
        - `"success"`: uses the ratio of successful larvae in each generation.
    - `dyn_metric`: string that determines how to aggregate the values of each substrate to get the metric of each. Possible values:
        - `"best"`: takes the best fitness
        - `"avg"`: takes the average fitness
        - `"med"`: takes the median fitness
        - `"worse"`: takes the worse fitness
    - `dyn_steps`: specifies the number of times the substrates will be evaluated. If `dyn_steps = -1`, the substrates will be evaluated every generation. 
    - `prob_amp`: float that determines how the differences between substrate metrics affect the probability of each one. A lower value means more amplification
- Stopping conditions (you only need to include the ones that will be used!):
    - `Neval`: number of evaluations of the fitness function
    - `Ngen`: number of generations
    - `time_limit`: execution time limit given in seconds (real time, not CPU time)
    - `fit_target`: value of the fitness function we want to reach
    - `stop_cond`: a string that determines the stopping condition. It can be simply the name of the criterion to be used (e.g. `Ngen`, `Neval`, etc), or also a logical expression that combines these different criteria (e.g. `Ngen or Neval`, `time_limit and fit_target`, etc). Must be included even if only a single criterion is used.
- Parallelization:
    - `Njobs`: the number of jobs to run in parallel. If `Njobs = 1`, the algorithm will run in sequential mode.
- Display options:
    - `verbose`: shows a periodic report of the algorithm's performance
    - `v_timer`: amount of time between each report

For examples of how to fill the parameters dictionary, please take a look at [main.py](/PyCROSL/main.py).

# Cite

If you use this implementation, please cite it as:

```bibtex
@article{perez2023,
  title={New Probabilistic, Dynamic Multi-Method Ensembles for Optimization Based on the CRO-SL},
  author={P{\'e}rez-Aracil, Jorge and Camacho-G{\'o}mez, Carlos and Lorente-Ramos, Eugenio and Marina, Cosmin M and Cornejo-Bueno, Laura M and Salcedo-Sanz, Sancho},
  journal={Mathematics},
  volume={11},
  number={7},
  pages={1666},
  year={2023},
  publisher={MDPI}
}
```

from PyCROSL.TestFunctions import *
from PyCROSL.CompareTests import *
from PyCROSL.CRO_SL import *
from PyCROSL.Substrate import *
from PyCROSL.SubstrateInt import *
from PyCROSL.SubstrateReal import *


def test_cro():
    DEparams = {"F":0.7, "Cr":0.8}
    substrates_int = [
        SubstrateInt("1point"),
        SubstrateInt("2point"),
        SubstrateInt("Multipoint"),
        SubstrateInt("WeightedAvg", {"F": 0.75}),
        SubstrateInt("BLXalpha", {"F": 0.5}),
        SubstrateInt("Multicross", {"N": 5}),
        SubstrateInt("Perm", {"Cr": 0.5}),
        SubstrateInt("Xor", {"Cr": 0.2}),
        SubstrateInt("MutNoise", {"method": "Gauss", "F":0.001, "N": 3}),
        SubstrateInt("MutSample", {"method": "Cauchy", "F":2, "N": 3}),
        SubstrateInt("RandNoise", {"method": "Laplace", "F":0.1}),
        SubstrateInt("RandSample", {"method": "Gauss", "F":0.1}),
        SubstrateInt("Gauss", {"F": 5}),
        SubstrateInt("Laplace", {"F": 5}),
        SubstrateInt("Cauchy", {"F": 10}),
        SubstrateInt("Poisson", {"F": 5}),
        SubstrateInt("Uniform", {"Low": -10, "Up": 10}),
        SubstrateInt("DE/rand/1", {"F":0.7, "Cr": 0.8}),
        SubstrateInt("DE/best/1", {"F":0.7, "Cr": 0.8}),
        SubstrateInt("DE/rand/2", {"F":0.7, "Cr": 0.8}),
        SubstrateInt("DE/best/2", {"F":0.7, "Cr": 0.8}),
        SubstrateInt("DE/current-to-rand/1", {"F":0.7, "Cr": 0.8}),
        SubstrateInt("DE/current-to-best/1", {"F":0.7, "Cr": 0.8}),
        SubstrateInt("DE/current-to-pbest/1", {"F":0.7, "Cr": 0.8, "P":0.11}),
        SubstrateInt("LSHADE", {"F":0.7, "Cr": 0.8, "P":0.11}),
        SubstrateInt("SA", {"F":5, "temp_ch": 20, "iter": 10}),
        SubstrateInt("HS", {"F":5, "Cr":0.3, "Par":0.1}),
        SubstrateInt("Dummy", {"F": 0.5})
    ]

    DEparams = {"F":0.7, "Cr":0.8}
    substrates_real = [
        SubstrateReal("1point"),
        SubstrateReal("2point"),
        SubstrateReal("Multipoint"),
        SubstrateReal("WeightedAvg", {"F": 0.75}),
        SubstrateReal("BLXalpha", {"F": 0.5}),
        SubstrateReal("SBX", {"Cr": 0.5}),
        SubstrateReal("Multicross", {"N": 5}),
        SubstrateReal("Perm", {"Cr": 0.5}),
        SubstrateReal("MutNoise", {"method": "Gauss", "F":5, "N": 3}),
        SubstrateReal("MutSample", {"method": "Cauchy", "F":2, "N": 3}),
        SubstrateReal("RandNoise", {"method": "Laplace", "F":6}),
        SubstrateReal("RandSample", {"method": "Gauss", "F":4}),
        SubstrateReal("Gauss", {"F":0.001}),
        SubstrateReal("Laplace", {"F":0.001}),
        SubstrateReal("Cauchy", {"F":0.002}),
        SubstrateReal("Uniform", {"Low": -10, "Up": 10}),
        SubstrateReal("DE/rand/1", {"F":0.7, "Cr": 0.8}),
        SubstrateReal("DE/best/1", {"F":0.7, "Cr": 0.8}),
        SubstrateReal("DE/rand/2", {"F":0.7, "Cr": 0.8}),
        SubstrateReal("DE/best/2", {"F":0.7, "Cr": 0.8}),
        SubstrateReal("DE/current-to-rand/1", {"F":0.7, "Cr": 0.8}),
        SubstrateReal("DE/current-to-best/1", {"F":0.7, "Cr": 0.8}),
        SubstrateReal("DE/current-to-pbest/1", {"F":0.7, "Cr": 0.8, "P":0.11}),
        SubstrateReal("LSHADE", {"F":0.7, "Cr": 0.8, "P":0.11}),
        SubstrateReal("SA", {"F":0.01, "temp_ch": 20, "iter": 10}),
        SubstrateReal("HS", {"F":0.01, "Cr":0.3, "Par":0.1}),
        SubstrateReal("Firefly", {"a":0.5, "b":1, "d":0.95, "g":10}),
        SubstrateReal("Dummy", {"F": 100})
    ]
    
    params = {
        "popSize": 500,
        "rho": 0.6,
        "Fb": 0.98,
        "Fd": 0.2,
        "Pd": 0.8,
        "k": 3,
        "K": 20,
        "group_subs": True,

        "stop_cond": "Ngen or Neval",
        "time_limit": 4000.0,
        "Ngen": 200,
        "Neval": 10e5,
        "fit_target": 1000,

        "verbose": True,
        "v_timer": 1,
        "Njobs": 1,

        "dynamic": True,
        "dyn_method": "success",
        "dyn_metric": "avg",
        "dyn_steps": 10,
        "prob_amp": 0.01
    }

    #objfunc = MaxOnes(1000, "min")
    #objfunc = MaxOnesReal(1000)
    # objfunc = Sphere(30)
    # objfunc = Rosenbrock(30)
    #objfunc = Katsuura(30)
    objfunc = Rastrigin(30)

    
    c = CRO_SL(objfunc, substrates_real, params)
    ind, fit = c.optimize()
    print(ind)
    c.display_report()

    c = CRO_SL(objfunc, substrates_int, params)
    c.restart() # reset the objective function counter
    ind, fit = c.optimize()
    print(ind)
    c.display_report()

def test_multi_cond():
    substrates_real = [
        SubstrateReal("1point"),
        SubstrateReal("Gauss", {"F":0.001}),
    ]

    params = {
        "popSize": 500,
        "rho": 0.6,
        "Fb": 0.98,
        "Fd": 0.2,
        "Pd": 0.8,
        "k": 3,
        "K": 20,
        "group_subs": True,

        "stop_cond": "time_limit or Ngen or Neval or fit_target",
        "time_limit": 9999,
        "Ngen": 9999,
        "Neval": 9999999,
        "fit_target": 0,

        "verbose": False,
        "v_timer": 1,
        "Njobs": 1,

        "dynamic": True,
        "dyn_method": "success",
        "dyn_metric": "avg",
        "dyn_steps": 10,
        "prob_amp": 0.01
    }

    # Testing three conditions: Ngen or fit_target or time_limit
    objfunc = Rastrigin(2)

    stopping_values = {"time_limit": 3, "Ngen": 200, "Neval": 1000, "fit_target": 0.1}
    for criterion_name, criterion_val in stopping_values.items():
        params_ = params.copy()
        params_[criterion_name] = criterion_val

        print('-'*50)
        print(f"Should stop with '{criterion_name}' reaching {criterion_val}")
        print('-'*50)

        c = CRO_SL(objfunc, substrates_real, params_)
        c.restart()
        _, fit = c.optimize()

        print(f"Generations: {len(c.history)}")
        print(f"Time spent: {round(c.real_time_spent, 3)}s")
        print(f"Best fitness: {fit}")
        print(f"Fitness evaluations: {c.objfunc.counter}")


def test_files():
    params = {
        "popSize": 100,
        "rho": 0.6,
        "Fb": 0.98,
        "Fd": 1,
        "Pd": 0.1,
        "k": 3,
        "K": 20,
        "group_subs": False,

        "stop_cond": "Neval",
        "time_limit": 4000.0,
        "Ngen": 3500,
        "Neval": 1e5,
        "fit_target": 1000,

        "verbose": True,
        "v_timer": 1,
        "Njobs": 1,

        "dynamic": True,
        "dyn_method": "fitness",
        "dyn_metric": "best",
        "dyn_steps": 100,
        "prob_amp": 0.001
    }

    c = CRO_SL(Sphere(100), [SubstrateReal("Gauss")], params)
    c.population.generate_random()
    c.save_solution()

def thirty_runs():
    DEparams = {"F":0.7, "Pr":0.8}
    substrates_real = [
        SubstrateReal("DE/rand/1", DEparams),
        SubstrateReal("DE/best/2", DEparams),
        SubstrateReal("DE/current-to-best/1", DEparams),
        SubstrateReal("DE/current-to-rand/1", DEparams)
    ]

    params = {
        "popSize": 100,
        "rho": 0.6,
        "Fb": 0.98,
        "Fd": 1,
        "Pd": 0.1,
        "k": 3,
        "K": 20,
        "group_subs": False,

        "stop_cond": "neval",
        "time_limit": 4000.0,
        "Ngen": 3500,
        "Neval": 1e5,
        "fit_target": 1000,

        "verbose": False,
        "v_timer": 1,
        "Njobs": 1,

        "dynamic": True,
        "dyn_method": "fitness",
        "dyn_metric": "avg",
        "dyn_steps": 100,
        "prob_amp": 0.1
    }

    n_coord = 30
    n_runs = 5
    
    combination_DE = [
        [0,1,2,3],
        [0,1,2],
        [0,1,3],
        [0,2,3],
        [1,2,3],
        [0,1],
        [0,2],
        [0,3],
        [1,2],
        [1,3],
        [2,3],
        [0],
        [1],
        [2],
        [3]
    ]

    for comb in combination_DE:
        substrates_filtered = [substrates_real[i] for i in range(4) if i in comb]
        fit_list = []
        for i in range(n_runs):
            c = CRO_SL(Rosenbrock(n_coord), substrates_filtered, params)
            _, fit = c.optimize_classic()
            fit_list.append(fit)
            #print(f"Run {i+1} {comb} ended")
            #c.display_report(show_plots=False)
        fit_list = np.array(fit_list)
        print(f"\nRESULTS {comb}:")
        print(f"min: {fit_list.min():e}; mean: {fit_list.mean():e}; std: {fit_list.std():e}")


def test_parallelism():
    jobs_list = [1, 2, 4, 8, 16]
    times_elapsed = []

    for n_jobs in jobs_list:
        print(f"Evaluating for {n_jobs} jobs")

        substrates_real = [
            SubstrateReal("Dummy", {"F": 100})
        ]

        params = {
            "popSize": 100,
            "rho": 0.6,
            "Fb": 0.98,
            "Fd": 0.2,
            "Pd": 0.8,
            "k": 3,
            "K": 20,
            "group_subs": True,

            "stop_cond": "Ngen",
            "time_limit": 4000.0,
            "Ngen": 20,
            "Neval": 10e5,
            "fit_target": 1000,

            "verbose": True,
            "v_timer": 1,
            "Njobs": n_jobs,

            "dynamic": True,
            "dyn_method": "success",
            "dyn_metric": "avg",
            "dyn_steps": 10,
            "prob_amp": 0.01
        }

        objfunc = TimeTest(10)

        c = CRO_SL(objfunc, substrates_real, params)
        c.optimize()

        times_elapsed.append(c.real_time_spent)

    for n_jobs, time_elapsed in zip(jobs_list, times_elapsed):
        print(f"{n_jobs} jobs: {time_elapsed:.2f}s")


def test_dist_params():
    objfunc = Sphere(5)
    objfunc.random_solution = lambda: np.full(objfunc.size, 200.0)
    substrates_real = [
        SubstrateReal("MutSample", {"method": "Uniform", "Low":np.array([40,10,2,7,30]), "Up":np.array([100,100,100,100,100]), "N": 2})
    ]

    params = {
        "popSize": 100,
        "rho": 0.6,
        "Fb": 0.98,
        "Fd": 0.2,
        "Pd": 0.8,
        "k": 3,
        "K": 20,
        "group_subs": True,

        "stop_cond": "Ngen",
        "Ngen": 500,

        "verbose": True,
        "v_timer": 1,
        "Njobs": 1,

        "dynamic": True,
        "dyn_method": "success",
        "dyn_metric": "avg",
        "dyn_steps": 10,
        "prob_amp": 0.01
    }
    
    c = CRO_SL(objfunc, substrates_real, params)
    ind, fit = c.optimize()
    print(ind)
    print()
    c.display_report(show_plots=False)

def main():
    # thirty_runs()
    # test_cro()
    # test_multi_cond()
    # test_files()
    # test_parallelism()
    test_dist_params()

if __name__ == "__main__":
    main()
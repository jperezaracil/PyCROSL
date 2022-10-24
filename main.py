from TestFunctions import *
from CompareTests import *
from CRO_SL import *
from Substrate import *
from SubstrateInt import *
from SubstrateReal import *


def test_cro():
    DEparams = {"F":0.7, "Cr":0.8}
    substrates_int = [
        SubstrateInt("1point"),
        SubstrateInt("2point"),
        SubstrateInt("Multipoint"),
        SubstrateInt("BLXalpha", {"Cr": 0.5}),
        SubstrateInt("Multicross", {"N": 5}),
        SubstrateInt("Perm", {"Cr": 0.5}),
        SubstrateInt("Xor", {"Cr": 0.1}),
        SubstrateInt("MutRand", {"method": "Gauss", "F":5, "Cr": 0.01}),
        SubstrateInt("Gauss", {"F": 5}),
        SubstrateInt("Laplace", {"F": 5}),
        SubstrateInt("Cauchy", {"F": 10}),
        SubstrateInt("Poisson", {"F": 5}),
        SubstrateInt("DE/rand/1", {"F":0.7, "Cr": 0.8}),
        SubstrateInt("DE/best/1", {"F":0.7, "Cr": 0.8}),
        SubstrateInt("DE/rand/2", {"F":0.7, "Cr": 0.8}),
        SubstrateInt("DE/best/2", {"F":0.7, "Cr": 0.8}),
        SubstrateInt("DE/current-to-rand/1", {"F":0.7, "Cr": 0.8}),
        SubstrateInt("DE/current-to-best/1", {"F":0.7, "Cr": 0.8}),
        SubstrateInt("DE/current-to-pbest/1", {"F":0.7, "Cr": 0.8}),
        SubstrateInt("LSHADE", {"F":0.7, "Cr": 0.8}),
        SubstrateInt("SA", {"F":5, "temp_ch": 20, "iter": 10}),
        SubstrateInt("HS", {"F":5, "Cr":0.3, "Par":0.1}),
        SubstrateInt("Replace", {"method":"Gauss", "F":0.1}),
        #SubstrateInt("Dummy", {"F": 0.5})
    ]

    DEparams = {"F":0.7, "Cr":0.8}
    substrates_real = [
        SubstrateReal("1point"),
        SubstrateReal("2point"),
        SubstrateReal("Multipoint"),
        SubstrateReal("BLXalpha", {"Cr": 0.5}),
        SubstrateReal("SBX", {"Cr": 0.5}),
        SubstrateReal("Multicross", {"N": 5}),
        SubstrateReal("Perm", {"Cr": 0.5}),
        SubstrateReal("MutRand", {"method": "Gauss", "F":0.01, "Cr": 0.01}),
        SubstrateReal("Gauss", {"F":0.001}),
        SubstrateReal("Laplace", {"F":0.001}),
        SubstrateReal("Cauchy", {"F":0.002}),
        SubstrateReal("DE/rand/1", {"F":0.7, "Cr": 0.8}),
        SubstrateReal("DE/best/1", {"F":0.7, "Cr": 0.8}),
        SubstrateReal("DE/rand/2", {"F":0.7, "Cr": 0.8}),
        SubstrateReal("DE/best/2", {"F":0.7, "Cr": 0.8}),
        SubstrateReal("DE/current-to-rand/1", {"F":0.7, "Cr": 0.8}),
        SubstrateReal("DE/current-to-best/1", {"F":0.7, "Cr": 0.8}),
        SubstrateReal("DE/current-to-pbest/1", {"F":0.7, "Cr": 0.8}),
        SubstrateReal("LSHADE", {"F":0.7, "Cr": 0.8}),
        SubstrateReal("SA", {"F":0.01, "temp_ch": 20, "iter": 10}),
        SubstrateReal("HS", {"F":0.01, "Cr":0.3, "Par":0.1}),
        SubstrateReal("Replace", {"method":"Gauss", "F":0.1}),
        SubstrateReal("Dummy", {"F": 100})
    ]
    
    params = {
        "popSize": 5000,
        "rho": 0.6,
        "Fb": 0.98,
        "Fd": 0.2,
        "Pd": 0.8,
        "k": 3,
        "K": 20,
        "group_subs": True,

        "stop_cond": "neval",
        "time_limit": 4000.0,
        "Ngen": 3500,
        "Neval": 10e5,
        "fit_target": 1000,

        "verbose": True,
        "v_timer": 1,

        "dynamic": True,
        "dyn_method": "success",
        "dyn_metric": "avg",
        "dyn_steps": 100,
        "prob_amp": 0.01
    }

    #objfunc = MaxOnes(1000, "min")
    #objfunc = MaxOnesReal(1000)
    #objfunc = Sphere(30)
    #objfunc = Rosenbrock(30)
    #objfunc = Katsuura(30)
    objfunc = Rastrigin(30)

    #c = CRO_SL(objfunc, substrates_int, params)
    c = CRO_SL(objfunc, substrates_real, params)
    ind, fit = c.optimize()
    print(ind)
    c.display_report()

def test_files():
    params = {
        "ReefSize": 100,
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

        "verbose": True,
        "v_timer": 1,

        "dynamic": True,
        "dyn_method": "fitness",
        "dyn_metric": "best",
        "dyn_steps": 100,
        "prob_amp": 0.001
    }

    c = CRO_SL(Sphere(100), [SubstrateReal("Gauss")], params)
    c.population.generate_random()
    c.save_solution()

def thity_runs():
    DEparams = {"F":0.7, "Pr":0.8}
    substrates_real = [
        SubstrateReal("DE/rand/1", DEparams),
        SubstrateReal("DE/best/2", DEparams),
        SubstrateReal("DE/current-to-best/1", DEparams),
        SubstrateReal("DE/current-to-rand/1", DEparams)
    ]

    params = {
        "ReefSize": 100,
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

def main():
    #thity_runs()
    test_cro()
    #test_files()

if __name__ == "__main__":
    main()
from CompareTests import Weierstrass
from CRO_SL import *

def test_cro():
    DEparams = {"F":0.7, "Pr":0.8}
    substrates_int = [
        #SubstrateInt("DE/best/1", {"F":0.8, "Pr":0.8}),
        #SubstrateInt("DE/rand/1", {"F":0.8, "Pr":0.8}),
        #SubstrateInt("DE/rand/2", {"F":0.8, "Pr":0.8}),
        #SubstrateInt("DE/best/2", {"F":0.8, "Pr":0.8}), 
        #SubstrateInt("DE/current-to-best/1", {"F":0.8, "Pr":0.8}),
        #SubstrateInt("DE/current-to-rand/1", {"F":0.8, "Pr":0.8}),
        #SubstrateInt("AddOne", {"F":0.1}),
        #SubstrateInt("DGauss", {"F":12}),
        #SubstrateInt("Perm", {"F":0.1}),
        #SubstrateInt("1point"),
        #SubstrateInt("2point"),
        #SubstrateInt("Multipoint"),
        SubstrateInt("Xor", {"F":0.002})
    ]

    DEparams = {"F":0.7, "Pr":0.8}
    substrates_real = [
        #SubstrateReal("SBX", {"F":0.8}),
        #SubstrateReal("Perm", {"F":0.6}),
        #SubstrateReal("1point"),
        #SubstrateReal("2point"),
        #SubstrateReal("Multipoint"),
        #SubstrateReal("Multicross", {"n_ind": 3}),
        #SubstrateReal("BLXalpha", {"F":0.8}),
        SubstrateReal("Replace", {"method": "Uniform", "F":1/3}),
        SubstrateReal("DE/best/1", DEparams),
        SubstrateReal("DE/rand/1", DEparams),
        SubstrateReal("DE/best/2", DEparams),
        SubstrateReal("DE/rand/2", DEparams),
        SubstrateReal("DE/current-to-best/1", DEparams),
        SubstrateReal("DE/current-to-rand/1", DEparams),
        SubstrateReal("LSHADE", {"F":0.5, "Pr":0.5}),
        #SubstrateReal("DE/current-to-rand/1", DEparams),
        #SubstrateReal("HS", {"F":0.5, "Pr":0.8}),
        #SubstrateReal("SA", {"F":0.14, "temp_ch":10, "iter":20}),
        #SubstrateReal("Cauchy", {"F":0.005}),
        #SubstrateReal("Cauchy", {"F":0.005}),
        #SubstrateReal("Gauss", {"F":0.5}),
        #SubstrateReal("Gauss", {"F":0.05}),
        #SubstrateReal("Gauss", {"F":0.001}),
        #SubstrateReal("DE/rand/1", DEparams),
        #SubstrateReal("DE/best/2", DEparams),
        #SubstrateReal("DE/current-to-pbest/1", DEparams),
        #SubstrateReal("DE/current-to-best/1", DEparams),
        #SubstrateReal("DE/current-to-rand/1", DEparams)
    ]
    
    params = {
        "ReefSize": 540,
        "rho": 0.7,
        "Fb": 0.98,
        "Fd": 0.8,
        "Pd": 0.2,
        "k": 7,
        "K": 3,
        "group_subs": True,

        "stop_cond": "neval",
        "time_limit": 40.0,
        "Ngen": 3500,
        "Neval": 3e5,
        "fit_target": 1000,

        "verbose": True,
        "v_timer": 1,

        "dynamic": True,
        "dyn_method": "success",
        "dyn_metric": "med",
        "dyn_steps": 75,
        "prob_amp": 0.05
    }

    #objfunc = MaxOnes(1000)
    #objfunc = MaxOnesReal(1000)
    #objfunc = Sphere(30)
    objfunc = Rosenbrock(30)
    #objfunc = Rastrigin(30)

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
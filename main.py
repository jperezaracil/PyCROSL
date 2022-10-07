from doctest import testfile
from CRO_SL import *

def test_cro():
    substrates_int = [
        #SubstrateInt("DE/best/1", {"F":0.8, "Pr":0.8}),
        #SubstrateInt("DE/rand/1", {"F":0.8, "Pr":0.8}),
        #SubstrateInt("DE/rand/2", {"F":0.8, "Pr":0.8}),
        #SubstrateInt("DE/best/2", {"F":0.8, "Pr":0.8}), 
        #SubstrateInt("DE/current-to-best/1", {"F":0.8, "Pr":0.8}),
        #SubstrateInt("DE/current-to-rand/1", {"F":0.8, "Pr":0.8}),
        #SubstrateInt("AddOne", {"F":0.1}),
        #SubstrateInt("DGauss", {"F":12}),
        SubstrateInt("Perm", {"F":0.1}),
        SubstrateInt("1point"),
        SubstrateInt("2point"),
        SubstrateInt("Multipoint"),
        SubstrateInt("Xor", {"F":0.002})
    ]

    DEparams = {"F":0.5, "Pr":0.8}
    substrates_real = [
        #SubstrateReal("SBX", {"F":0.5}),
        #SubstrateReal("Perm", {"F":0.3}),
        #SubstrateReal("1point"),
        #SubstrateReal("2point"),
        #SubstrateReal("Multipoint"),
        #SubstrateReal("BLXalpha", {"F":0.8}),
        #SubstrateReal("Rand"),
        #SubstrateReal("DE/best/1", {"F":0.6, "Pr":0.06}),
        #SubstrateReal("DE/rand/1", {"F":0.7, "Pr":0.25}),
        #SubstrateReal("DE/best/2", {"F":0.7, "Pr":0.25}),
        #SubstrateReal("DE/rand/2", {"F":0.6, "Pr":0.06}),
        #SubstrateReal("DE/current-to-best/1", {"F":0.7, "Pr":0.25}),
        #SubstrateReal("DE/current-to-rand/1", {"F":0.7, "Pr":0.25}),
        #SubstrateReal("HS", {"F":0.5, "Pr":0.8}),
        #SubstrateReal("SA", {"F":0.14, "temp_ch":10, "iter":20}),
        #SubstrateReal("Gauss", {"F":0.04}),

        #SubstrateReal("DE/rand/1", DEparams),
        SubstrateReal("DE/best/2", DEparams),
        #SubstrateReal("DE/current-to-best/1", DEparams),
        #SubstrateReal("DE/current-to-rand/1", DEparams)
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

        "verbose": True,
        "v_timer": 1,

        "dynamic": True,
        "dyn_method": "fitness",
        "dyn_metric": "best",
        "dyn_steps": 100,
        "prob_amp": 0.001
    }

    #objfunc = MaxOnes(1000)
    #objfunc = MaxOnesReal(1000)
    objfunc = Sphere(30)
    #objfunc = Test1(30)
    #objfunc = Rosenbrock(30)
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

        "dynamic": False,
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
    thity_runs()
    #test_cro()
    #test_files()

if __name__ == "__main__":
    main()
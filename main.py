from CRO_SL.CRO_SL import *
from Operator import *
from OperatorInt import *
from OperatorReal import *
from TestFunctions import *

def test_cro():
    operators_int = [
        #OperatorInt("DE/best/1", {"F":0.8, "Pr":0.8}),
        #OperatorInt("DE/rand/1", {"F":0.8, "Pr":0.8}),
        #OperatorInt("DE/rand/2", {"F":0.8, "Pr":0.8}),
        #OperatorInt("DE/best/2", {"F":0.8, "Pr":0.8}), 
        #OperatorInt("DE/current-to-best/1", {"F":0.8, "Pr":0.8}),
        #OperatorInt("DE/current-to-rand/1", {"F":0.8, "Pr":0.8}),
        #OperatorInt("AddOne", {"F":0.1}),
        #OperatorInt("DGauss", {"F":12}),
        OperatorInt("Perm", {"F":0.1}),
        OperatorInt("1point"),
        OperatorInt("2point"),
        OperatorInt("Multipoint"),
        OperatorInt("Xor", {"F":0.002})
    ]

    DEparams = {"F":0.5, "Pr":0.8}
    operators_real = [
        #OperatorReal("SBX", {"F":0.5}),
        #OperatorReal("Perm", {"F":0.3}),
        #OperatorReal("1point"),
        #OperatorReal("2point"),
        #OperatorReal("Multipoint"),
        #OperatorReal("BLXalpha", {"F":0.8}),
        #OperatorReal("Rand"),
        #OperatorReal("DE/best/1", {"F":0.6, "Pr":0.06}),
        #OperatorReal("DE/rand/1", {"F":0.7, "Pr":0.25}),
        #OperatorReal("DE/best/2", {"F":0.7, "Pr":0.25}),
        #OperatorReal("DE/rand/2", {"F":0.6, "Pr":0.06}),
        #OperatorReal("DE/current-to-best/1", {"F":0.7, "Pr":0.25}),
        #OperatorReal("DE/current-to-rand/1", {"F":0.7, "Pr":0.25}),
        #OperatorReal("HS", {"F":0.5, "Pr":0.8}),
        #OperatorReal("SA", {"F":0.14, "temp_ch":10, "iter":20}),
        #OperatorReal("Gauss", {"F":0.04}),

        #OperatorReal("DE/rand/1", DEparams),
        #OperatorReal("DE/best/2", DEparams),
        OperatorReal("DE/current-to-best/1", DEparams)#,
        #OperatorReal("DE/current-to-rand/1", DEparams)
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

    objfunc = MaxOnes(1000)
    #objfunc = MaxOnesReal(1000)
    #objfunc = Sphere(30, "min")
    #objfunc = Test1(30)
    #objfunc = Rosenbrock(30)
    #objfunc = Rastrigin(30)

    c = CRO_SL(objfunc, operators_int, params)
    #c = CRO_SL(objfunc, operators_real, params)
    ind, fit = c.optimize()
    print(ind)
    c.display_report()

def thity_runs():
    DEparams = {"F":0.7, "Pr":0.8}
    operators_real = [
        OperatorReal("DE/rand/1", DEparams),
        OperatorReal("DE/best/2", DEparams),
        OperatorReal("DE/current-to-best/1", DEparams),
        OperatorReal("DE/current-to-rand/1", DEparams)
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
        "Neval": 3e5,
        "fit_target": 1000,

        "verbose": False,
        "v_timer": 1,

        "dynamic": False,
        "dyn_method": "fitness",
        "dyn_metric": "best",
        "dyn_steps": 100,
        "prob_amp": 0.1
    }

    n_coord = 30
    n_runs = 10
    
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
        operators_filtered = [operators_real[i] for i in range(4) if i in comb]
        fit_list = []
        for i in range(n_runs):
            c = CRO_SL(Rosenbrock(n_coord), operators_filtered, params)
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

if __name__ == "__main__":
    main()
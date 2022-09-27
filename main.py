from CRO_SL import *

def test_cro():
    substrates_int = [
        SubstrateInt("DE/best/1"),
        SubstrateInt("DE/rand/1"),
        SubstrateInt("DE/rand/2"),
        SubstrateInt("DE/best/2"), 
        SubstrateInt("DE/current-to-best/1"),
        SubstrateInt("DE/current-to-rand/1"),
        SubstrateInt("AddOne"),
        SubstrateInt("DGauss"),
        SubstrateInt("Perm"),
        SubstrateInt("1point"),
        SubstrateInt("2point"),
        SubstrateInt("Multipoint"),
        SubstrateInt("Xor")
    ]

    substrates_real = [
        #SubstrateReal("SBX", {"F":0.5}),
        #SubstrateReal("Perm", {"F":0.3}),
        #SubstrateReal("1point"),
        #SubstrateReal("2point"),
        #SubstrateReal("Multipoint"),
        #SubstrateReal("BLXalpha", {"F":0.8}),
        #SubstrateReal("Rand"),
        #SubstrateReal("DE/best/1", {"F":0.5, "Pr":0.8}),
        SubstrateReal("DE/rand/1", {"F":0.8, "Pr":0.7}),
        SubstrateReal("DE/best/2", {"F":0.8, "Pr":0.8}),
        #SubstrateReal("DE/rand/2", {"F":0.5, "Pr":0.8}),
        SubstrateReal("DE/current-to-best/1", {"F":0.8, "Pr":0.8}),
        SubstrateReal("DE/current-to-rand/1", {"F":0.8, "Pr":0.7}),
        #SubstrateReal("HS", {"F":0.5, "Pr":0.8}),
        #SubstrateReal("SA", {"F":0.14, "temp_ch":10, "iter":20}),
        #SubstrateReal("Gauss", {"F":0.04})
    ]
    
    params = {
        "ReefSize": 300,
        "rho": 0.6,
        "Fd": 0.2,
        "Pd": 0.1,
        "k": 3,
        "K": 13,

        "stop_cond": "neval",
        "time_limit": 120.0,
        "Ngen": 200,
        "Neval": 3e5,
        "fit_target": 1000,

        "verbose": True,
        "v_timer": 1,

        "dynamic": False,
        "dyn_metric": "avg",
        "prob_amp": 0.1
    }

    #objfunc = MaxOnes(1000)
    #objfunc = MaxOnesReal(1000)
    #objfunc = Sphere(1000)
    #objfunc = Test1(30)
    objfunc = Rosenbrock(30)
    #objfunc = Rastrigin(30)

    #c = CRO_SL(objfunc, substrates_int, params)
    
    c = CRO_SL(objfunc, substrates_real, params)
    ind, fit = c.optimize()
    print(ind)
    c.display_report()

def thity_runs():
    substrates_real = [
        SubstrateReal("DE/rand/1", {"F":0.6, "Pr":0.7}),
        SubstrateReal("DE/best/2", {"F":0.8, "Pr":0.8}),
        SubstrateReal("DE/current-to-best/1", {"F":0.8, "Pr":0.8}),
        SubstrateReal("DE/current-to-rand/1", {"F":0.6, "Pr":0.7})
    ]

    params = {
        "ReefSize": 100,
        "rho": 0.8,
        "Fd": 0.1,
        "Pd": 0.1,
        "k": 3,
        "K": 5,

        "stop_cond": "neval",
        "time_limit": 120.0,
        "Ngen": 200,
        "Neval": 3e4,
        "fit_target": 1000,

        "verbose": False,
        "v_timer": 3,

        "dynamic": True,
        "dyn_metric": "max",
        "prob_amp": 0.01
    }

    n_coord = 30
    fit_list = []
    for i in range(30):
        c = CRO_SL(Rosenbrock(n_coord), substrates_real, params)
        _, fit = c.optimize()
        fit_list.append(-fit)
        print(-fit)
        print(f"Run {i+1} done")
    fit_list = np.array(fit_list)
    print(fit_list.min(), fit_list.mean(), fit_list.std())

def main():
    #thity_runs()
    test_cro()

if __name__ == "__main__":
    main()
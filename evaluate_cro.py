import signal

import numpy as np
import pandas as pd
from CompareTests import *
from CRO_SL import CRO_SL
from SubstrateReal import *


def exec_runs(evalg_inst, dframe, func_name, subs_name, nruns=10):
    fit_list = []
    for i in range(nruns):
        _, fit = evalg_inst.optimize_classic()
        evalg_inst.restart()
        fit_list.append(fit)
    fit_list = np.array(fit_list)
    print(f"\nRESULTS {func_name}-{subs_name}:")
    print(f"min: {fit_list.min():e}; mean: {fit_list.mean():e}; std: {fit_list.std():e}")
    dframe.loc[len(dframe)] = [func_name, subs_name, fit_list.min(), fit_list.mean(), fit_list.std()]

def save_dframe_error(dframe, last_func, last_subs, filename):
    save_dframe(dframe, filename)
    with open("cro_log.txt", "w") as f:
        f.write("Warning: stopped before all the runs were completed")
        f.write(f"\t - last function: {last_func}, with substrates: {last_subs}")

def save_dframe(dframe, filename):
    dframe.to_csv(filename)
    print("saved dataframe to disk")

def do_on_shutdown(sig, fr, exec_data, last_func_name, comb_name):
    print("saving before shutdown")
    save_dframe_error(exec_data, last_func_name, comb_name)
    exit(1)

def main():
    successful_file = "cro_classic_results.csv"
    error_file = "cro_classic_error.csv"


    DEparams = {"F":0.7, "Pr":0.8}
    substrates_real = [
        SubstrateReal("DE/rand/1", DEparams),
        SubstrateReal("DE/best/2", DEparams),
        SubstrateReal("DE/current-to-best/1", DEparams),
        SubstrateReal("DE/current-to-rand/1", DEparams)
    ]

    combination_DE = [
        [0,1,2,3],
        [0,1,2],[0,1,3],[0,2,3],[1,2,3],
        [0,1],[0,2],[0,3],[1,2],[1,3],[2,3],
        [0],[1],[2],[3]
    ]

    params = {
        "popSize": 100,
        "rho": 0.6,
        "Fb": 0.98,
        "Fd": 0.1,
        "Pd": 0.9,
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
        "dyn_method": "success",
        "dyn_metric": "avg",
        "dyn_steps": 75,
        "prob_amp": 0.1
    }

    funcs = [
        Sphere(30),
        HighCondElliptic(30),
        BentCigar(30),
        Discus(30),
        Rosenbrock(30),
        Ackley(30),
        Weierstrass(30),
        Griewank(30),
        Rastrigin(30),
        ModSchwefel(30),
        Katsuura(30),
        HappyCat(30),
        HGBat(30),
        ExpandedGriewankPlusRosenbrock(30),
        ExpandedShafferF6(30)
    ]

    exec_data = pd.DataFrame(columns = ["function", "operator", "best", "mean", "std"])
    last_func_name = str(type(funcs[0]))[8:-2].replace("CompareTests.", "")
    comb_name = "4_DE0123"

    # THIS LINE ONLY WORKS ON LINUX execute function on shutdown
    signal.signal(signal.SIGTERM, lambda sig, fr: do_on_shutdown(sig, fr, exec_data, last_func_name, comb_name))

    try:
        for f in funcs:
            last_func_name = str(type(f))[8:-2].replace("CompareTests.", "")
            print()
            print(last_func_name)

            for comb in combination_DE:
                comb_name = f"{len(comb)}DE_{''.join([str(i) for i in comb])}"
                substrates_filtered = [substrates_real[i] for i in range(4) if i in comb]
                c = CRO_SL(f, substrates_filtered, params)
                exec_runs(c, exec_data, last_func_name, comb_name, nruns=10)
        save_dframe(exec_data, successful_file)
    except KeyboardInterrupt as k:
        print("\nexecution stopped early")
        save_dframe_error(exec_data, last_func_name, comb_name, error_file)
    except Exception as e:
        print(f"\nsomething went wrong:\n{e}")
        save_dframe_error(exec_data, last_func_name, comb_name, error_file)

if __name__ == "__main__":
    main()
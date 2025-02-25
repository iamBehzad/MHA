from imports import *

funcs = opfunu.get_functions_based_classname("2015")[:5]
benchmarks = []

# Initialize the problems dictionary
problems = []
for i, benchmark in enumerate(funcs):
    f = benchmark(30, f_bias=0)
    benchmarks.append(f"F{i + 1}")
    problems.append({
        "bounds": FloatVar(lb=f.lb, ub=f.ub),
        "obj_func": f.evaluate,
        "minmax": "min",
        "name": f"F{i + 1}",
        "log_to": None, # "console",
    })

# Define models (PSO, WOA, GA, DE, AOA, SCSO)
models = [
    ade(epoch=100, pop_size=30),
    PSO.OriginalPSO(epoch=100, pop_size=30),
    WOA.OriginalWOA(epoch=100, pop_size=30),
    GA.BaseGA(epoch=100, pop_size=30),
    DE.OriginalDE(epoch=100, pop_size=30),
    AOA.OriginalAOA(epoch=100, pop_size=30),
    SCSO.OriginalSCSO(epoch=100, pop_size=30),
]

# Define termination criteria
term = {
    "max_fe": 3000
}

# Run Multitask for all benchmarks
if __name__ == "__main__":
    multitask = Multitask(
        algorithms=models,
        problems=problems,
        terminations=(term,),
        modes=("single",),  # Modes can be adjusted as needed
        n_workers=4
    )

    multitask.execute(n_trials=10, n_jobs=6, save_path="history", save_as="csv", save_convergence=True, verbose=False)
    
    stats_df = read_algorithm_data(BEST_FIT_FOLDER, ALGORITHMS)
    performance_df = prepare_performance_data(stats_df)
    friedman_results = perform_friedman_test(stats_df.to_dict('records'))
    wilcoxon_results = perform_wilcoxon_test(stats_df.to_dict('records'), PROPOSED_ALGORITHM)
    print("save_to_excel")
    save_to_excel(performance_df, friedman_results, wilcoxon_results, RESULT_FILENAME)

    convergence_data = read_convergence_data(CONVERGENCE_FOLDER, benchmarks, ALGORITHMS)
    plot_convergence_curves(convergence_data, benchmarks, ALGORITHMS, OUTPUT_FOLDER)
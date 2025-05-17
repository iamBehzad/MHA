from helpers import *
funcs = opfunu.get_functions_based_classname("2017")
benchmarks = []

# Initialize the problems dictionary
for i, benchmark in enumerate(funcs):
    f = benchmark(30, f_bias=0)
    benchmarks.append(f"F{i + 1}")
    
    
stats_df = read_algorithm_data(BEST_FIT_FOLDER, ALGORITHMS)
performance_df = prepare_performance_data(stats_df)
friedman_results = perform_friedman_test(stats_df.to_dict('records'))
wilcoxon_results = perform_wilcoxon_test(stats_df.to_dict('records'), PROPOSED_ALGORITHM)
print("save_to_excel")
save_to_excel(performance_df, friedman_results, wilcoxon_results, RESULT_FILENAME)

convergence_data = read_convergence_data(CONVERGENCE_FOLDER, benchmarks, ALGORITHMS)
plot_convergence_curves(convergence_data, benchmarks, ALGORITHMS, OUTPUT_FOLDER)
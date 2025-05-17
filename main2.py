from imports import *
from ade import *
from arde import *
from helpers import *

TCSP = TensionCompressionSpringProblem
CBP = CantileverBeamProblem
SRP = SpeedReducerProblem
CSIP = CarSideImpactProblem
PVP = PressureVesselProblem
TBTP = ThreeBarTrussProblem
GTP = GearTrainProblem
IBP = IBeamProblem
TCP = TubularColumnProblem
PLP = PistonLeverProblem
CBP = CorrugatedBulkheadProblem
WBP = WeldedBeamProblem
RCBP = ReinforcedConcreateBeamProblem
funcs = [TCSP, CBP, SRP, CSIP, PVP, TBTP, GTP, IBP, TCP, PLP, CBP, WBP, RCBP]
TCSP.n_objs
benchmarks = ['TensionCompressionSpringProblem', 'CantileverBeamProblem', 'SpeedReducerProblem', 'CarSideImpactProblem', 'PressureVesselProblem', 
              'ThreeBarTrussProblem', 'GearTrainProblem', 'IBeamProblem', 'TubularColumnProblem', 'PistonLeverProblem', 'CorrugatedBulkheadProblem',
              'WeldedBeamProblem', 'ReinforcedConcreateBeamProblem']

# Initialize the problems dictionary
problems = []
for i, benchmark in enumerate(funcs):
    #f = benchmark(30, f_bias=0)
    f = benchmark(30)
    #benchmarks.append(f"F{i + 1}")
    problems.append({
        "bounds": FloatVar(lb=f.lb, ub=f.ub),
        "obj_func": f.evaluate,
        "obj_weights" : f.n_objs * [1],
        "minmax": "min",
        "name": benchmarks[i],
        "log_to": None, # "console",
    })

# Define models (PSO, WOA, GA, DE, AOA, SCSO)
models = [
    ARDE(epoch=500, pop_size=30),
    PSO.OriginalPSO(epoch=500, pop_size=30),
    WOA.OriginalWOA(epoch=500, pop_size=30),
    GA.BaseGA(epoch=500, pop_size=30),
    DE.OriginalDE(epoch=500, pop_size=30),
    AOA.OriginalAOA(epoch=500, pop_size=30),
    SCSO.OriginalSCSO(epoch=500, pop_size=30),
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
        #terminations=(term,),
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
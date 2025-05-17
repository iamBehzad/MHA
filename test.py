from imports import *
from fno import *

objective_function = opfunu.get_functions_by_classname("F42005")
objective_function = objective_function[0](ndim=10)

problem_dict1 = {
    "obj_func": objective_function.evaluate,
    "bounds": FloatVar(lb=objective_function.lb, ub=objective_function.ub),
    "minmax": "min",
    "save_population": "true",
    "log_to": None
}

epoch = 10
pop_size = 30
model = fno(epoch, pop_size)
#model= GWO.OriginalGWO(epoch, pop_size)
g_best = model.solve(problem_dict1)
print(f"FNO Fitness: {g_best.target.fitness}")

model = WOA.OriginalWOA(epoch, pop_size)
#model= GWO.OriginalGWO(epoch, pop_size)
g_best = model.solve(problem_dict1)
print(f"WOA Fitness: {g_best.target.fitness}")
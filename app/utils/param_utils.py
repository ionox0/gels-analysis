from itertools import product


def create_params_combs_dicts(param_grid):
    params_set = [param_grid[key] for key in param_grid.keys()]
    params_combs = list(product(*params_set))
    params_combs_dicts = [dict(zip(param_grid.keys(), p)) for p in params_combs]
    return params_combs_dicts
from .solver import Solver

def solver_entry(C):
    return globals()[C.config["common"]["solver"]["type"]](C)

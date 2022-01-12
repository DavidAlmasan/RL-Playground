import sys, os
from os.path import join


CUR = os.path.abspath(os.path.dirname(__file__))

###################################
# Name of cfg used for experiment #
###################################
cfg_name = 'a2c'

if cfg_name == 'a2c':
    from base_algorithms.config.a2c import get_config

cfg = get_config()
# Instantiate the solver
if cfg.SOLVER.ASYNC:
    raise NotImplementedError(f'Async solver not ported to torch yet')
    from solvers.async_solver import AsyncSolver
    solver = AsyncSolver(cfg)
else:
    if cfg.SOLVER.TYPE == 'dqn':
        from solvers.dqn_solver import DQNSolver
        solver = DQNSolver(cfg)
    elif cfg.SOLVER.TYPE == 'ddqn':
        from solvers.ddqn_solver import DDQNSolver
        solver = DDQNSolver(cfg)
    elif cfg.SOLVER.TYPE == 'a2c':
        from solvers.a2c_solver import A2CSolver
        solver = A2CSolver(cfg)
    else:
        raise NotImplementedError('TODO')

# Train
# solver.train()

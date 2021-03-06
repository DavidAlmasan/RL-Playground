import sys, os
from os.path import join

from config.base_config import cfg, cfg_from_file
from solvers.dqn_solver import DQNSolver
from solvers.ddqn_solver import DDQNSolver
from solvers.a2c_solver import A2CSolver
from solvers.async_solver import AsyncSolver


CUR = os.path.abspath(os.path.dirname(__file__))

###################################
# Name of cfg used for experiment #
###################################
cfg_name = 'a2c.yaml'

# Parse new config into default one
if cfg_name is not None:
    cfg_path = join(*[CUR, 'config', cfg_name])
    cfg_from_file(cfg_path)

# Instantiate the solver
if cfg.ASYNC.USE:
    solver = AsyncSolver(cfg)
else:
    if cfg.TYPE == 'dqn':
        solver = DQNSolver(cfg)
    elif cfg.TYPE == 'ddqn':
        solver = DDQNSolver(cfg)
    elif cfg.TYPE == 'a2c':
        solver = A2CSolver(cfg)
    else:
        raise NotImplementedError('TODO')

# Train
solver.train()

import sys, os
from os.path import join

from config.base_config import cfg, cfg_from_file
from solvers.dqn_solver import DQNSolver
from solvers.ddqn_solver import DDQNSolver


CUR = os.path.abspath(os.path.dirname(__file__))

###################################
# Name of cfg used for experiment #
###################################
cfg_name = 'ddqn.yaml'

# Parse new config into default one
if cfg_name is not None:
    cfg_path = join(*[CUR, 'config', cfg_name])
    cfg_from_file(cfg_path)

# Instantiate the solver
if cfg.TYPE == 'dqn':
    solver = DQNSolver(cfg)
elif cfg.TYPE == 'ddqn':
    solver = DDQNSolver(cfg)
else:
    raise NotADirectoryError('TODO')

# Train
solver.train()

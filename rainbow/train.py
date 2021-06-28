"""Top level script used to train an agent using Rainbow framework"""
import warnings

from rainbow.config import get_config
from rainbow.solver import RainbowSolver


def main(debug):
    config = get_config()
    solver = RainbowSolver(config)
    solver.solve()
    print('Finished.')

if __name__ == "__main__":
    debug = True
    if debug:
        warnings.warn('Script will run in debug mode.')
    main(debug)
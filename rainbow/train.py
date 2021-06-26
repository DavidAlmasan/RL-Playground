"""Top level script used to train an agent using Rainbow framework"""
import warnings

from config import get_config
from solver import Solver


def main(debug):
    config = get_config()
    solver = Solver(config)
    solver.solve()
    print('Finished.')

if __name__ == "__main__":
    debug = True
    if debug:
        warnings.warn('Script will run in debug mode.')
    main(debug)
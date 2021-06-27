import warnings
from rainbow.train import main  # Use rainbow training

if __name__ == "__main__":
    debug = True
    if debug:
        warnings.warn('Script will run in debug mode.')
    main(debug)
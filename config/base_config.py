import os
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Implemented models
TYPES = ['dqn', 'ddqn', 'a2c']

# Name, save path and weights path
cfg.NAME = 'experiment_1'
cfg.TYPE = 'dqn'

# Env
cfg.ENV = edict()
cfg.ENV.NAME = 'CartPole-v0'

# Model
cfg.MODEL = edict()
cfg.MODEL.TYPE = 'ffn'  # plan to use ffn, conv, seq
cfg.MODEL.ARCH = [512, 256, 64]
cfg.MODEL.LOSS = 'mse'  # plan to use mse, huber cross-entropy
cfg.MODEL.LOAD_FILE = ''
cfg.MODEL.INIT = 'xavier'  # xavier or gauss

# Model misc
cfg.MODEL.MISC = edict()
cfg.MODEL.MISC.DUELING = False   # Dueling architecture

# Hyperparameters
cfg.HYPERPARAMS = edict()
cfg.HYPERPARAMS.LEARNING_RATE = 0.00025
cfg.HYPERPARAMS.GAMMA = 1.
cfg.HYPERPARAMS.GAMMA_FACTOR = 0.999
cfg.HYPERPARAMS.BATCH_SIZE = 32

# Train params
cfg.TRAIN = edict()
cfg.TRAIN.OPTIMIZER = 'adam'
cfg.TRAIN.MAX_STEPS = 500
cfg.TRAIN.MAX_EPISODES = 200
cfg.TRAIN.WAIT_EPISODES = 10
cfg.TRAIN.MEMORY_LEN = 200000


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k in a:
        # a must specify keys that are in b
        v = a[k]
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)

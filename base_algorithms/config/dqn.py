from ml_collections import ConfigDict


def get_config():
    C = ConfigDict()

    # Experiment
    C.EXPERIMENT = ConfigDict()
    # C.EXPERIMENT.NAME = 'Breakout-V4'
    C.EXPERIMENT.NAME = 'foo'
    C.EXPERIMENT.SUFFIX = 'trial1'

    # Solver
    C.SOLVER = ConfigDict()
    C.SOLVER.TYPE = 'dqn'
    C.SOLVER.ASYNC = False

    # Environment
    C.ENV = ConfigDict()
    # C.ENV.NAME = 'Breakout-v4'
    C.ENV.NAME = 'CartPole-v0'

    # Data
    C.DATA = ConfigDict()
    # C.DATA.INPUT_SHAPE = (84, 84)
    C.DATA.PREPROCESS = []

    # Agent
    C.AGENT = ConfigDict()
    ### Allowed models: ['resnet18', 'small_cnn', 'medium_cnn', 'small_mlp', 'medium_mlp', 'other']
    C.AGENT.MODEL = 'other'  # other expects a custom model with parameters defined in KWARGS
    C.AGENT.KWARGS = ConfigDict()
    C.AGENT.KWARGS.TYPE = 'ffn'
    C.AGENT.KWARGS.POLICY_TYPE = C.SOLVER.TYPE
    C.AGENT.KWARGS.ARCHITECTURE = [512, 256, 64]

    C.AGENT.HYPERPARAMS = ConfigDict()
    C.AGENT.HYPERPARAMS.LEARNING_RATE = 1e-4
    C.AGENT.HYPERPARAMS.GAMMA = 1.0
    C.AGENT.HYPERPARAMS.EPS_REDUCTION_FACTOR = 0.99
    C.AGENT.HYPERPARAMS.NUM_FRAMES_PER_SAMPLE = 4

    # Train
    C.TRAIN = ConfigDict()
    ### Allowed optimizers: ['adam', 'sgd', 'rmsprop', 'radam']
    C.TRAIN.OPTIMIZER = 'adam'
    C.TRAIN.LOSS = 'mse'
    C.TRAIN.MAX_STEPS_EPISODE = 5000
    C.TRAIN.MAX_STEPS_VALIDATION_EPISODE = 1000
    C.TRAIN.MAX_EPISODES = 1000
    C.TRAIN.MEMORY_LEN = 10
    C.TRAIN.REPLAY_PERIOD = 4
    C.TRAIN.EVAL_PERIOD = 4 * 100
    C.TRAIN.BATCH_SIZE = 32
    C.TRAIN.TOT_TRAIN_STEPS = 100000

    return C




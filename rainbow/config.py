from ml_collections import ConfigDict

def get_config():
    C = ConfigDict()

    # Experiment
    C.EXPERIMENT = ConfigDict()
    ### Allowed names: ['Breakout-v4', 'Breakout-v0']
    C.EXPERIMENT.NAME = 'Breakout-v4'
    C.EXPERIMENT.SUFFIX = 'trial1'

    C.DATA = ConfigDict()
    C.DATA.INPUT_SHAPE = (84, 84)
    # Agent
    C.AGENT = ConfigDict()
    ### Allowed models: ['resnet18', 'small_cnn', 'medium_cnn', 'small_mlp', 'medium_mlp', 'other']
    C.AGENT.MODEL = 'small_cnn' # other expects a class named OtherAgent from a module called other_model.py
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
    C.TRAIN.MAX_EPISODES = 10
    C.TRAIN.REPLAY_PERIOD = 4
    C.TRAIN.EVAL_PERIOD = 4 * 100
    C.TRAIN.BATCH_SIZE = 32
    C.TRAIN.TOT_TRAIN_STEPS = 100000

    # Rainbow params
    C.RAINBOW = ConfigDict()
    C.RAINBOW.TARGET_NET_COPY_TRAINSTEPS = 50
    C.RAINBOW.DOUBLE_DQN = False 
    
    # Replay buffer 
    C.RAINBOW.REPLAY_BUFFER = ConfigDict()
    C.RAINBOW.REPLAY_BUFFER.MEMORY_LEN = 1e5
    C.RAINBOW.REPLAY_BUFFER.ALPHA = 0.5
    C.RAINBOW.REPLAY_BUFFER.BETA = 0.5

    return C


    

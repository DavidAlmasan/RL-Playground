from ml_collections import ConfigDict

def get_config():
    C = ConfigDict()

    # Experiment
    C.EXPERIMENT = ConfigDict()
    ### Allowed names: ['BypedalWalker-v2', 'Breakout-v0']
    C.EXPERIMENT.NAME = 'Breakout-v0'
    C.EXPERIMENT.SUFFIX = 'trial1'

    # Agent
    C.AGENT = ConfigDict()
    ### Allowed models: ['resnet18', 'small_cnn', 'medium_cnn', 'small_mlp', 'medium_mlp', 'other']
    C.AGENT.MODEL = 'small_cnn' # other expects a class named OtherAgent from a module called other_model.py
    C.AGENT.HYPERPARAMS = ConfigDict()
    C.AGENT.HYPERPARAMS.LEARNING_RATE = 1e-4
    C.AGENT.HYPERPARAMS.GAMMA = 1.0

    # Train
    C.TRAIN = ConfigDict()
    ### Allowed optimizers: ['adam', 'sgd', 'rmsprop', 'radam']
    C.TRAIN.OPTIMIZER = 'adam'
    C.TRAIN.MAX_STEPS_EPISODE = 1000 
    C.TRAIN.MAX_EPISODES = 10
    C.TRAIN.WAIT_EPISODES = 0
    C.TRAIN.MEMORY_LEN = 10000
    C.TRAIN.BATCH_SIZE = 2

    # Rainbow params
    C.RAINBOW = ConfigDict()

    return C


    

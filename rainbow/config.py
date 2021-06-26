from ml_collections import ConfigDict

def get_config():
    C = ConfigDict()

    # Experiment
    C.EXPERIMENT = ConfigDict()
    ### Allowed names: ['BypedalWalker-v2', 'Breakout-v0']
    C.EXPERIMENT.NAME = 'CarRacing-v0'

    # Agent
    C.AGENT = ConfigDict()
    ### Allowed models: ['resnet18', 'small_cnn', 'medium_cnn', 'small_mlp', 'medium_mlp', 'other']
    C.AGENT.MODEL = 'other' # other expects a class named OtherAgent from a module called other_model.py
    C.AGENT.HYPERPARAMS = ConfigDict()
    ### Allowed optimizers: ['adam', 'sgd', 'rmsprop', 'radam']
    C.AGENT.HYPERPARAMS.OPTIMIZER = 'adam'
    C.AGENT.HYPERPARAMS.LEARNING_RATE = 1e-4
    C.AGENT.HYPERPARAMS.GAMMA = 1.0

    # Train
    C.TRAIN = ConfigDict()
    C.TRAIN.MAX_STEPS_EPISODE = 1000 
    C.TRAIN.MAX_EPISODES = 10
    C.TRAIN.WAIT_EPISODES = 0
    C.TRAIN.MEMORY_LEN = 10000

    # Rainbow params
    C.RAINBOW = ConfigDict()


    

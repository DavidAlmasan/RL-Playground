"""
Implements Misc classes needed within RL software paradigm
    EnvWrapper
    ActionSpace

Implements the BaseSolver class for optimising all NN models through RL
    BaseSolver
"""
import sys, os
from shutil import rmtree
from os.path import join
from PIL import Image
import numpy as np
from collections import deque
from termcolor import *
import colorama
from loguru import logger

import gym
import torch
import torch.optim
import torch.nn
from torchvision import transforms

from base_algorithms.utils.utils import bottom_n_percent, play_tf

colorama.init()
# Some globals
CUR = os.path.abspath(os.path.dirname(__file__))
ALLOWED_ENVIRONMENTS = ['Breakout-v0', 'Breakout-v4', 'CartPole-v0']
ALLOWED_OPTIMIZERS = ['adam', 'sgd', 'rmsprop']
ALLOWED_MODELS = ['resnet18', 'small_cnn', 'medium_cnn', 'small_mlp', 'medium_mlp', 'other']


class ActionSpace:
    """
    Base class used in EnvWrapper to mimic openai.gym calls
    """
    def __init__(self, actions):
        self.actions = actions
        self.sample_space = list(range(2 * len(actions) + 1))

    def sample(self):
        return np.random.choice(self.sample_space)


class EnvWrapper:
    """
    Base class for parsing general torch DataLoaders/Datasets
    using openai.gym-type calls
    """
    def __init__(self, dataloader, **kwargs):  # more like dataset but anyway
        self.dataloader = dataloader
        self.train_idx = 0
        self.eval_idx = 0
        self.max_l = len(self.dataloader)
        self.action_space = kwargs['action_space']

    def return_prices(self):
        """
        Returns the the BASE/QUOTE bid price at market value
        """
        if isinstance(self.dataloader, torch.utils.data.Dataset):
            return self.dataloader.return_prices()
        else:
            return self.dataloader.dataset.return_prices()

    def train(self):
        """
        Sets the env to train subset
        """
        self.dataloader.train()
        self.max_l = len(self.dataloader)

    def eval(self):
        """
        Sets the env to validation subset
        """
        self.dataloader.eval()
        self.max_l = len(self.dataloader)

    def step(self, action, funds):
        """
        Openai.gym-type call:
        Steps into Env state s with action

        Returns: next state, reward, is done, misc info
        """
        info = {}
        if self.dataloader.train:
            self.train_idx += 1
            s_ = self.dataloader[self.train_idx % self.max_l]

            if self.train_idx == self.max_l - 1:
                d = 1
            else:
                d = 0

        else:
            self.eval_idx += 1
            s_ = self.dataloader[self.eval_idx % self.max_l]

            if self.eval_idx == self.max_l - 1:
                d = 1
            else:
                d = 0

        if d:
            info['balance'] = {'base': 0., 'quote': 0.}
            return s_, 0., d, info
        if self.dataloader.train:
            r, balance = self.dataloader.create_reward(action, self.train_idx % self.max_l, funds)
        else:
            r, balance = self.dataloader.create_reward(action, self.eval_idx % self.max_l, funds)
        info['balance'] = balance
        return s_, r, d, info

    def reset(self):
        """
        Openai.gym-type call:
        Resets train or validation subset to starting state

        Returns starting state
        """
        if self.dataloader.train:
            self.train_idx = 0

        else:
            self.eval_idx = 0
        return self.dataloader[0]


class BaseSolver:
    """
    Base class for optimising all NN models through RL

    Two different init functions based on use case:

    base_init: Used to run experiments using only parameters/models defined cfg file.
               Used in general purpose RL research experiments

    agent_init: Used to run experiments with custom agents and dataloaders


    """
    def __init__(self, cfg, agent=None, dataloader=None, **kwargs):
        # Preprocessing
        self.preprocess_dict = {
            'nparray_to_pil': Image.fromarray,
            'np_squeeze': np.squeeze,
            'gray3': transforms.Grayscale(3),
            'gray1': transforms.Grayscale(1),
        }

        if agent is None:
            assert dataloader is None, 'Dataloader should also be None if agent is None'
            self.custom_init = False
            self.base_init(cfg)
        else:
            assert dataloader is not None, 'Dataloader should also NOT be None if agent is not None'
            self.custom_init = True
            self.agent_init(cfg, agent, dataloader, kwargs['save_path'])
        self.train_iter = 0



    def agent_init(self, cfg, agent, dataloader, save_path):
        """
        Agent and dataloader objects are passed as argument to solver.
        """
        self.cfg = cfg
        self.save_path = save_path
        self.logger = logger
        self.logger.add(join(self.save_path, 'log.log'), format="{time} {level} {message}", level="INFO")

        # Hyperparams
        self.learning_rate = self.cfg.HYPERPARAMS.LEARNING_RATE
        self.gamma = self.cfg.HYPERPARAMS.GAMMA
        self.eps_reduction_factor = self.cfg.HYPERPARAMS.EPS_REDUCTION_FACTOR

        # Training params
        self.batch_size = self.cfg.TRAIN.BATCH_SIZE
        self.max_steps_per_episode = self.cfg.TRAIN.MAX_STEPS_EPISODE
        self.max_steps_per_valepisode = self.cfg.TRAIN.MAX_STEPS_VALIDATION_EPISODE
        self.validation_iters = self.cfg.TRAIN.VALIDATION_ITERS
        try:
            self.memory = deque(maxlen=self.cfg.TRAIN.MEMORY_LEN)
        except:
            self.memory = None # specific module implenmentation of replay buffer 

        # Env, agent, optimizer and loss
        self.env, self.agent, self.optimizer = None, None, None
        self.env = self.wrap_dataloader(dataloader, action_space=ActionSpace(self.cfg.ACTION_SPACE))

        self.agent = agent
        self.target_agent = agent  # TODO: might need copy.deepcopy()

        if self.cfg.OPTIMIZER in ALLOWED_OPTIMIZERS:
            self.optimizer = self.create_optimizer(cfg.OPTIMIZER, self.agent.parameters())
        else:
            raise NotImplementedError('Optimizer name: {} not in {}'.format(self.cfg.TRAIN.OPTIMIZER, ALLOWED_OPTIMIZERS))

        self.criterion = self.build_loss(cfg.TRAIN.LOSS)

        # Misc
        self.best_val_score = -np.inf

    def wrap_dataloader(self, dataloader, action_space):
        """
        Returns a openai.gym-type EnvWrapper for a given dataloader
        """
        return EnvWrapper(dataloader, action_space=action_space)

    def base_init(self, cfg):
        """
        Used for general purpose Single Agent RL (SARL) experiments
        """
        self.cfg = cfg
        self.agent_cfg = self.cfg.AGENT
        # Hyperparams
        self.learning_rate = self.agent_cfg.HYPERPARAMS.LEARNING_RATE
        self.gamma = self.agent_cfg.HYPERPARAMS.GAMMA
        self.eps_reduction_factor = self.agent_cfg.HYPERPARAMS.EPS_REDUCTION_FACTOR

        # Data transforms
        if 'PREPROCESS' in self.cfg.DATA.keys():
            self.transforms = []
            for p in self.cfg.DATA.PREPROCESS:
                # Does not include transforms with keywords or totensor
                t = self.preprocess_dict[p]
                self.transforms.append(t)

            if 'INPUT_SHAPE' in self.cfg.DATA.keys():
                self.transforms.append(transforms.Resize(self.cfg.DATA.INPUT_SHAPE))
                self.transforms.append(transforms.ToTensor())
            else:
                self.transforms.append(torch.tensor)
        else:
            self.transforms = [transforms.Resize(self.cfg.DATA.INPUT_SHAPE),
                                                  transforms.Grayscale(),
                                                  transforms.ToTensor()]

        # Training params
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_steps_per_episode = cfg.TRAIN.MAX_STEPS_EPISODE
        self.max_steps_per_valepisode = cfg.TRAIN.MAX_STEPS_VALIDATION_EPISODE
        self.max_episodes = cfg.TRAIN.MAX_EPISODES
        self.log_episodes = [int(float(self.max_episodes) * i / 10) for i in range(11)]
        try:
            self.memory = deque(maxlen=cfg.TRAIN.MEMORY_LEN)
        except:
            self.memory = None  # specific module implementation of replay buffer

        # Env, agent, optimizer, loss and misc
        self.env, self.val_env = None, None
        self.agent, self.target_agent = None, None
        self.criterion, self.optimizer = None, None
        self.name, self.save_path, self.weights_path = None, None, None

        if cfg.ENV.NAME in ALLOWED_ENVIRONMENTS:
            self.env = self.create_env(cfg.ENV.NAME)
            self.val_env = self.create_env(cfg.ENV.NAME)
        else:
            raise NotImplementedError('Environment name: {} not in {}'.format(cfg.ENV.NAME, ALLOWED_ENVIRONMENTS))
        
        if self.agent_cfg.MODEL in ALLOWED_MODELS:
            self.agent = self.create_agent(self.agent_cfg)
            self.target_agent = self.create_agent(self.agent_cfg)
        else:
            raise NotImplementedError('Model name: {} not in {}'.format(self.agent_cfg.MODEL, ALLOWED_MODELS))

        if cfg.TRAIN.OPTIMIZER in ALLOWED_OPTIMIZERS:
            self.optimizer = self.create_optimizer(cfg.TRAIN.OPTIMIZER, self.agent.parameters())
        else:
            raise NotImplementedError('Optimizer name: {} not in {}'.format(cfg.TRAIN.OPTIMIZER, ALLOWED_OPTIMIZERS))

        self.criterion = self.build_loss(cfg.TRAIN.LOSS)

        # Misc
        self.logger = logger
        self.save_path = join('experiments/', cfg.EXPERIMENT.NAME + '_' + cfg.EXPERIMENT.SUFFIX)
        while os.path.isdir(self.save_path):
            self.logger.warning(f'Experiment exists at path: {self.save_path}. Appending <+> to save name...')
            self.save_path += '+'
        self.weights_path = join(self.save_path, 'weights')
        try:
            os.makedirs(self.weights_path)
        except:
            self.logger.warning(f'Experiment exists at path: {self.save_path}. Exiting...')
            os.makedirs(self.weights_path)
        self.name = cfg.EXPERIMENT.NAME



        # Load weights
        if "LOAD_FILE" in self.agent_cfg.keys():
            self.logger.info('Loading weights from file: {}'.format(self.agent_cfg.LOAD_FILE))
            self.agent.load_weights(self.agent_cfg.LOAD_FILE)

    def build_loss(self, loss_type):
        """
        Returns a loss function object
        """
        if loss_type == 'mse':
            loss = torch.nn.MSELoss(reduction='none')
        elif loss_type == 'huber':
            loss = torch.nn.HuberLoss(reduction='none')
        else:
            raise NotImplementedError('TODO')

        return loss

    def preprocess(self, state):
        """
        Base preprocessing function for Environment states
        Particular implementation by Child Solver classes
        """
        for t in self.transforms:
            state = t(state)
        return state.type(torch.FloatTensor)

    def validate_agent(self, games, num_steps, render, multihead=False):
        """
        Base function for evaluating agent
        Particular implementation by Child Solver classes
        TODO: Change to be general purpose (i.e. from steps to some validation metric defined
              by Solver and Env type
        """
        self.agent.eval()
        steps = []
        perc = 1
        for _ in range(games):
            steps.append(self.play(num_steps, render))
        self.logger.info('Average reward of {} games : {}'.format(games, colored(np.mean(steps), 'green')))
        self.logger.info('STD of reward of {} games : {}'.format(games, colored(float("{:.2f}".format(np.std(steps))),
                                                                         'green')))
        bot_mean = bottom_n_percent(steps, perc)
        self.logger.info('Average reward of bottom {}% games : {}'.format(perc, colored(bot_mean, 'yellow')))
        self.logger.info('-----------------------------------')
        self.agent.train()
        return np.mean(steps), float("{:.2f}".format(np.std(steps))), bot_mean

    def update_agent_weights(self, epoch=None):
        """
        Will load model with the latest saved weights
        TODO: Port to pytorch
        """
        ckpt_file = tf.train.latest_checkpoint(self.weights_path)
        if epoch is not None:
            ckpt_file = join(self.weights_path, '{}-epoch_'.format(self.name) + str(epoch) + '.ckpt')
        self.logger.info('Loading weights from file: {}'.format(ckpt_file))
        self.agent.load_weights(ckpt_file)

    def create_env(self, env_name):
        """
        Creates an openai.gym environment given env name
        """
        env = gym.make(env_name)
        if self.max_steps_per_episode is not None:
            env._max_episode_steps = self.max_steps_per_episode
        env.reset()
        return env

    def get_agent(self):
        return self.agent

    def get_env(self):
        return self.env

    def save_agent(self, path):
        """
        Saves the agent weights at a given epoch
        """
        torch.save(self.agent.state_dict(), path)

    def create_optimizer(self, optimizer_type, params):
        """
        Returns optimizer object by optimizer_type
        """
        if optimizer_type == 'adam':
            return torch.optim.Adam(params, self.learning_rate)
        elif optimizer_type == 'rmsprop':
            return torch.optim.RMSprop(params, self.learning_rate)
        elif optimizer_type == 'sgd':
            return torch.optim.SGD(params, self.learning_rate)
        else:
            raise NotImplementedError(f'Optimizer name: {optimizer_type} not implemented')

    def create_agent(self, agent_config):
        """
        Returns an agent model given an agent config
        """
        # Get input shape
        proxy_env = self.create_env(self.cfg.ENV.NAME)
        s = proxy_env.reset()
        action_space_length = proxy_env.action_space.n
        del proxy_env
        # Agent
        if agent_config.MODEL == "resnet18":
            raise NotImplementedError(f'Model name: resnet18 not implemented')
        elif agent_config.MODEL == 'small_cnn':
            from rainbow.model import SmallCNN as Agent
        elif agent_config.MODEL == 'other':
            agent = self.generate_model_from_cfg(agent_config.KWARGS, s.shape, action_space_length)
            return agent
        else:
            raise NotImplementedError(f'Model name: {agent_config.MODEL} not implemented')
        
        agent = Agent(action_space_length)
        return agent

    def generate_model_from_cfg(self, params, state_space_shape, action_space_length):
        """
        Returns a nn.Model agent defined by the params (agent_config.KWARGS)
        Expects a KWARGS dictionary containing all the necessary info to generate the model
        FFN models with single dim input spaces supported for now,
        when KWARGS changes, will adapt this function accordingly
        """
        if params.TYPE == 'ffn':
            if len(state_space_shape) != 1:
                raise NotImplementedError(f'state space ndim({len(state_space_shape)}) > 1')
            if params.POLICY_TYPE == 'a2c':
                from base_algorithms.models.ffn import ActorCritic
                # Ignore Batch dimension of state_space_shape
                agent = ActorCritic(state_space_shape[-1], params.ARCHITECTURE, action_space_length)
                return agent
            elif params.POLICY_TYPE == 'dqn':
                from base_algorithms.models.ffn import SimpleFFN
                # Ignore Batch dimension of state_space_shape
                agent = SimpleFFN(state_space_shape[-1], params.ARCHITECTURE, action_space_length)
                return agent

            else:
                raise NotImplementedError(f'Policy TYPE {params.POLICY_TYPE} not currently supported')

        else:
            raise NotImplementedError(f'Model TYPE {params.TYPE} not currently supported')

    def epsilon_greedy(self, s):
        """
        Epsilon-greedy strategy for action selection at time t, given state s
        """
        eps = max(0.1, self.eps_reduction_factor ** self.train_iter)
        eps = float("{:.2f}".format(eps))
        s = self.preprocess(s)
        action_space = self.agent(s)
        if eps <= np.random.uniform(0, 0.9999):
            # Perform explore action
            action = self.env.action_space.sample()
        else:
            # Perform greedy action
            action = torch.argmax(action_space, dim=-1).numpy()
        return action, eps

    def remember(self, state, action, reward, next_state, done):
        """
        Appends training sample to Memory replay buffer
        """
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        self.logger.info('Implementation depends on type of solver used')

    def train_step(self, agent):
        self.logger.info('Implementation depends on type of solver used')

    def stopping_criterion(self):
        if self.train_iter >= self.max_steps_per_episode:
            return True
        return False

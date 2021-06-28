import sys, os
from os.path import join
import numpy as np
from collections import deque
from termcolor import *
import colorama

import gym
import torch
import torch.optim
import torch.nn
from torchvision import transforms

from base_algorithms.utils.utils import bottom_n_percent, play

colorama.init()
# Some globals
CUR = os.path.abspath(os.path.dirname(__file__))
ALLOWED_ENVIRONMENTS = ['Breakout-v0', 'Breakout-v4']
ALLOWED_OPTIMIZERS = ['adam', 'sgd', 'rmsprop']
ALLOWED_MODELS = ['resnet18', 'small_cnn', 'medium_cnn', 'small_mlp', 'medium_mlp', 'other']

class BaseSolver():
    def __init__(self, cfg):
        self.cfg = cfg
        self.agent_cfg = self.cfg.AGENT
        # Hyperparams
        self.learning_rate = self.agent_cfg.HYPERPARAMS.LEARNING_RATE
        self.gamma = self.agent_cfg.HYPERPARAMS.GAMMA
        self.eps_reduction_factor = self.agent_cfg.HYPERPARAMS.EPS_REDUCTION_FACTOR

        # Data transforms
        self.transforms = transforms.Compose([transforms.Resize(self.cfg.DATA.INPUT_SHAPE),
                                              transforms.Grayscale(),
                                              transforms.ToTensor()])

        # Training params
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_steps_per_episode = cfg.TRAIN.MAX_STEPS_EPISODE
        self.max_episodes = cfg.TRAIN.MAX_EPISODES
        self.log_episodes = [int(float(self.max_episodes) * i / 10) for i in range(11)]
        self.memory = deque(maxlen=cfg.TRAIN.MEMORY_LEN)

        # Env, agent, optimizer and loss
        self.env, self.agent, self.optimizer = None, None, None
        if cfg.EXPERIMENT.NAME in ALLOWED_ENVIRONMENTS:
            self.env = self.create_env(cfg.EXPERIMENT.NAME)
        else:
            raise NotImplementedError('Experiment name: {} not in {}'.format(cfg.EXPERIMENT.NAME, ALLOWED_ENVIRONMENTS))
        
        if self.agent_cfg.MODEL in ALLOWED_MODELS:
            self.agent = self.create_agent(self.agent_cfg)
        else:
            raise NotImplementedError('Model name: {} not in {}'.format(self.agent_cfg.MODEL, ALLOWED_MODELS))

        if cfg.TRAIN.OPTIMIZER in ALLOWED_OPTIMIZERS:
            self.optimizer = self.create_optimizer(cfg.TRAIN.OPTIMIZER, self.agent.parameters())
        else:
            raise NotImplementedError('Optimizer name: {} not in {}'.format(cfg.TRAIN.OPTIMIZER, ALLOWED_OPTIMIZERS))

        # self.criterion = self.build_loss(cfg.MODEL.LOSS)

        # Misc
        self.save_path = 'experiments/' + cfg.EXPERIMENT.SUFFIX + '_' + cfg.EXPERIMENT.NAME + '.json'
        self.weights_path = join('weights', cfg.EXPERIMENT.NAME)
        self.name = cfg.EXPERIMENT.NAME



        # Load weights
        if "LOAD_FILE" in self.agent_cfg.keys():
            print('Loading weights from file: {}'.format(self.agent_cfg.LOAD_FILE))
            self.agent.load_weights(self.agent_cfg.LOAD_FILE)

    def build_loss(self, loss_type):  # TODO change this to pytorch
        if loss_type == 'mse':
            loss = torch.nn.MSELoss()
        elif loss_type == 'huber':
            loss = torch.nn.HuberLoss()
        else:
            raise NotImplementedError('TODO')

        return loss

    def preprocess(self, state):
        return self.transforms(state)

    def validate_agent(self, multihead=False):
        steps = []
        games = 100
        perc = 1
        for _ in range(games):
            steps.append(play(self.env, self.agent, 200, False, multihead))
        print('Average timesteps of {} games : {}'.format(games, colored(np.mean(steps), 'green')))
        print('STD of  timesteps of {} games : {}'.format(games, colored(float("{:.2f}".format(np.std(steps))),
                                                                         'green')))
        bot_mean = bottom_n_percent(steps, perc)
        print('Average timesteps of bottom {}% games : {}'.format(perc, colored(bot_mean, 'yellow')))
        print('-----------------------------------')
        return np.mean(steps), float("{:.2f}".format(np.std(steps))), bot_mean

    def update_agent_weights(self, epoch=None):
        ckpt_file = tf.train.latest_checkpoint(self.weights_path)
        if epoch is not None:
            ckpt_file = join(self.weights_path, '{}-epoch_'.format(self.name) + str(epoch) + '.ckpt')
        print('Loading weights from file: {}'.format(ckpt_file))
        self.agent.load_weights(ckpt_file)

    def create_env(self, env_name):
        env = gym.make(env_name)
        if self.max_steps_per_episode is not None:
            env._max_episode_steps = self.max_steps_per_episode
        env.reset()
        return env

    def get_agent(self):
        return self.agent

    def get_env(self):
        return self.env

    def save_agent(self, path, epoch):
        if self.weights_path is not None:
            os.makedirs(self.weights_path, exist_ok=True)
            path = join(path, '{}-epoch_{}.ckpt'.format(self.name, epoch))
            self.agent.save_weights(path)

    def create_optimizer(self, optimizer_type, params):
        # Optimizer
        if optimizer_type == 'adam':
            return torch.optim.Adam(params, self.learning_rate)
        elif optimizer_type == 'rmsprop':
            return torch.optim.RMSprop(params, self.learning_rate)
        elif optimizer_type == 'sgd':
            return torch.optim.SGD(params, self.learning_rate)
        else:
            raise NotImplementedError('TODO')

    def create_agent(self, agent_config):
        # Get input shape
        proxy_env = self.create_env(self.cfg.EXPERIMENT.NAME)
        s = proxy_env.reset()
        action_space_length = proxy_env.action_space.n
        del proxy_env
        # Agent
        if agent_config.MODEL == "resnet18":
            raise NotImplementedError('TODO')
        elif agent_config.MODEL == 'small_cnn':
            from rainbow.model import SmallCNN as Agent
        else:
            raise NotImplementedError('TODO')
        
        agent = Agent(action_space_length)
        return agent

    def epsilon_greedy(self, t, s):
        eps = max(0.1, self.eps_reduction_factor ** t)
        eps = float("{:.2f}".format(eps))
        # s = self.preprocess(s)
        action_space = self.agent(s)
        if eps <= np.random.uniform(0, 0.9999):
            # Perform explore action
            action = self.env.action_space.sample()
        else:
            # Perform greedy action
            action = torch.argmax(action_space, dim=-1)
        return action, eps

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        print('Implementation depends on type of solver used')

    def train_step(self, agent):
        print('Implementation depends on type of solver used')


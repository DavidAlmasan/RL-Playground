import sys, os
from os.path import join
import numpy as np
from collections import deque
from termcolor import *
import colorama

import gym
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import MeanSquaredError, Huber

from utils.utils import bottom_n_percent, play

colorama.init()
CUR = os.path.abspath(os.path.dirname(__file__))


class BaseSolver():
    def __init__(self, cfg):
        # Hyperparams
        self.learning_rate = cfg.HYPERPARAMS.LEARNING_RATE
        self.gamma = cfg.HYPERPARAMS.GAMMA
        self.batch_size = cfg.HYPERPARAMS.BATCH_SIZE
        self.eps_red_factor = 0.999

        # Training params
        self.max_steps_per_episode = cfg.TRAIN.MAX_STEPS
        self.max_episodes = cfg.TRAIN.MAX_EPISODES
        self.wait_episodes = cfg.TRAIN.WAIT_EPISODES
        self.log_episodes = [int(float(self.max_episodes) * i / 10) for i in range(11)]
        self.memory = deque(maxlen=cfg.TRAIN.MEMORY_LEN)

        # Env, agent, optimizer and loss
        self.arch = cfg.MODEL.ARCH
        self.agent_misc = cfg.MODEL.MISC
        self.env = self.create_env(cfg.ENV.NAME)
        self.agent, self.optimizer = None, None
        self.agent = self.create_agent(cfg.MODEL.TYPE,
                                       cfg.MODEL.ARCH,
                                       cfg.MODEL.MISC)

        self.criterion = self.build_loss(cfg.MODEL.LOSS)

        # Misc
        self.save_path = cfg.LOGPATH
        self.weights_path = join('weights', cfg.WEIGHTS)
        self.name = cfg.NAME
        self.cfg = cfg

    def build_loss(self, loss_type):
        if loss_type == 'mse':
            return MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        elif loss_type == 'huber':
            return Huber(reduction=tf.keras.losses.Reduction.SUM)
        else:
            raise NotImplementedError('TODO')

    def preprocess(self, state):
        return np.expand_dims(state, axis=0)

    def validate_agent(self):
        steps = []
        games = 100
        perc = 1
        for _ in range(games):
            steps.append(play(self.env, self.agent, 200, False))
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

    def create_optimizer(self, optimizer_type):
        # Optimizer
        if optimizer_type == 'adam':
            return Adam(self.learning_rate)
        elif optimizer_type == 'rmsprop':
            return RMSprop(self.learning_rate)
        else:
            raise NotImplementedError('TODO')

    def create_agent(self, agent_type, arch, misc):
        # Agent
        if agent_type == 'ffn':
            from models.ffn import Agent
            ag = Agent(arch, self.env.action_space.n, misc.DUELING)
            return ag

        else:
            raise NotImplementedError('TODO')

    def epsilon_greedy(self, t, s):
        eps = max(0.1, self.eps_red_factor ** t)
        eps = float("{:.2f}".format(eps))
        s = self.preprocess(s)
        action_space = np.squeeze(self.agent(s).numpy())
        if eps <= np.random.uniform(0, 0.9999):
            # Perform explore action
            action = self.env.action_space.sample()
        else:
            # Perform greedy action
            action = np.argmax(action_space)
        return action, eps

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        print('Implementation depends on type of solver used')

    def train_step(self, agent, target_agent):
        print('Implementation depends on type of solver used')


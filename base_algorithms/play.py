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

env_name = 'CartPole-v0'
env = gym.make(env_name)
env.reset()
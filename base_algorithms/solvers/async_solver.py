"""
Implements solver for All Asynchronous model solvers
NOTE: if model is ActorCritic, Solver becomes A3C
TODO: port to pytorch
"""

import sys, os
from os.path import join
import numpy as np
from collections import deque
from termcolor import *
import colorama
import threading
from copy import deepcopy
from itertools import accumulate
import json


import gym
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import MeanSquaredError, Huber

from utils.utils import bottom_n_percent, play

from solvers.base_solver import BaseSolver

colorama.init()
CUR = os.path.abspath(os.path.dirname(__file__))


# TODO barebones implementation. need proper training loop and
# TODO upload/download of weights
class AsyncSolver(BaseSolver):
    def __init__(self, cfg):
        super(AsyncSolver, self).__init__(cfg)
        self.num_threads = cfg.ASYNC.THREADS
        self.download_cycle = cfg.ASYNC.DOWNLOAD_CYCLE
        self.upload_cycle = cfg.ASYNC.UPLOAD_CYCLE
        self.validate_cycle = cfg.ASYNC.VALIDATE_CYCLE
        self.lock = threading.Lock()
        self.num_updates = 0
        self.best_score = 0.

        self.json_path = join(CUR, self.save_path)
        self.log_path = self.json_path[:-4] + 'txt'

    # def validate_shared_model(self):
    #     self.lock.acquire()
    #     n = deepcopy(self.num_updates)
    #     self.lock.release()
    #
    #     if n % self.validate_cycle == 0 :
    #         self.validate_agent(multihead=True)

    def train_local_agent(self):
        if self.cfg.TYPE == 'a2c':
            from solvers.a2c_solver import A2CSolver
            local_actor = A2CSolver(self.cfg)
        elif self.cfg.TYPE == 'dqn':
            from solvers.dqn_solver import DQNSolver
            local_actor = DQNSolver(self.cfg)
        elif self.cfg.TYPE == 'ddqn':
            from solvers.ddqn_solver import DDQNSolver
            local_actor = DDQNSolver(self.cfg)

        for ep in range(self.max_episodes):
            # noinspection PyUnboundLocalVariable
            grads = local_actor.train(async_solver=True)
            if (ep + 1) % self.download_cycle == 0 and ep + 1 >= self.download_cycle:
                # noinspection PyUnboundLocalVariable
                self.download_weights(local_actor.agent)
            if (ep + 1) % self.upload_cycle == 0 and ep + 1 >= self.upload_cycle:
                self.upload_grads(grads)

    def train(self):

        with open(self.log_path, 'w') as loss_file:
            loss_file.write('Metrics:' + '\n')

        with open(self.json_path, 'w') as json_file:
            json.dump(self.cfg, json_file, sort_keys=True,
                      indent=4, separators=(',', ': '))

        threads = list()
        for thr_idx in range(self.num_threads):
            thr = threading.Thread(target=self.train_local_agent,
                                   name='local_agent'+str(thr_idx))
            threads.append(thr)
        # Thread to validate the shared model
        # val_thr = threading.Thread(target=self.validate_shared_model, args=(True, ))
        # threads.append(val_thr)

        for thr in threads:
            thr.start()
        for thr in threads:
            thr.join()
        print('Finished Async Training')


    def download_weights(self, local_model):
        """
        Downloads weights from the shared model into the actor_learner
        :return: None
        """
        # Download weights
        self.lock.acquire()
        shared_weights = self.agent.get_weights()
        self.lock.release()

        # Apply update
        local_model.set_weights(shared_weights)

    def upload_grads(self, grads):
        """
        Uploads weights of thr actor_learner to the shared model
        :return: None
        """

        # Apply new weights to shared model
        self.lock.acquire()
        self.num_updates += 1
        self.optimizer.apply_gradients(zip(grads, self.agent.trainable_variables))
        if self.num_updates >= self.validate_cycle and self.num_updates % self.validate_cycle == 0:
            mu, std, bot_mu = self.validate_agent(multihead=True)
            if mu > self.best_score:
                self.best_score = mu
                self.save_agent(self.weights_path, 'best')
        self.lock.release()


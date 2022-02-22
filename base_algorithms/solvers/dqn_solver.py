"""
Implements DQN solver based on BaseSolver
TODO: port to pytorch
      hint: draw inspiration from trading repo solvers
"""
import random
import os
from os.path import join
import json
import numpy as np
import collections

import torch

from .base_solver import BaseSolver

CUR = os.path.abspath(os.path.dirname(__file__))


class DQNSolver(BaseSolver):
    """
    Child Solver class for training models using the DQN algorithm
        Playing Atari with Deep Reinforcement Learning: https://arxiv.org/pdf/1312.5602.pdf
    """
    def __init__(self, cfg, agent=None, dataloader=None, **kwargs):
        super(DQNSolver, self).__init__(cfg, agent, dataloader, **kwargs)

        self.ddqn_prob = 0

    def train(self):
        """
        Top level training function
        """
        itx = 0
        best_score = 0.
        json_path = join(self.save_path, 'metrics.json')
        log_path = json_path[:-4] + 'txt'

        with open(log_path, 'w') as loss_file:
            loss_file.write('Metrics:' + '\n')

        for episode in range(self.max_episodes):
            loss = 0.
            s = self.env.reset()
            for t in range(self.max_steps_per_episode):
                # Epsilon greedy with eps = 1/(itx+1)
                action, eps = self.epsilon_greedy(s)
                s_, r, d, _ = self.env.step(action)
                if d:
                    r = -100.
                self.remember(self.preprocess(s),
                              action,
                              r,
                              self.preprocess(s_),
                              d)
                s = s_
                if np.random.uniform(0, 1) > self.ddqn_prob:
                    loss += self.train_step(self.agent, self.target_agent)
                else:
                    loss += self.train_step(self.target_agent, self.agent)

                if d:
                    loss = float("{:.2f}".format(loss / (t + 1)))
                    self.logger.info(f'Episode length: {t + 1} with final eps: {eps}, and loss {loss}')
                    break

                    # mu, std, bot_mu = self.validate_agent()
                    log_str = 'mean: {}, std: {}, bottom_mean: {}, loss: {}'.format(mu, std, bot_mu, loss)
                    if mu > best_score:
                        best_score = mu
                        self.save_agent(self.weights_path, 'best')
                    # if only dqn, save weights from agent to target_agent
                    if self.ddqn_prob == 0:
                        agent_weights = self.agent.get_weights()
                        self.target_agent.set_weights(agent_weights)

                    with open(log_path, 'a') as loss_file:
                        loss_file.write(log_str + '\n')
                    break
            if episode in self.log_episodes:
                weights_save_path = join(self.weights_path, str(episode) + '.pt')
                self.save_agent(weights_save_path)

        print('Finished training!')

    def train_step(self, agent, target_agent):
        """
        Implements gradient based optimisation step for agent drawing samples from the Memory buffer
        """
        agent.train()
        target_agent.eval()

        if isinstance(self.memory, collections.deque):
            if len(self.memory) < self.batch_size:
                return 0.
        else:
            raise NotImplementedError('Replay buffer only available in Rainbow')

        minibatch = random.sample(self.memory, self.batch_size)
        state = self.preprocess(torch.stack([minibatch[i][0][0] for i in range(self.batch_size)]))
        action = torch.tensor([minibatch[i][1] for i in range(self.batch_size)])
        reward = torch.tensor([minibatch[i][2] for i in range(self.batch_size)])
        next_state = self.preprocess(torch.stack([minibatch[i][3][0] for i in range(self.batch_size)]))
        done = torch.tensor([minibatch[i][4] for i in range(self.batch_size)])

        with torch.no_grad():
            q_values_t, _ = agent(state)
            q_values_t1 = torch.argmax(agent(next_state)[0], dim=-1)

            target_agent_q_values_s_ = target_agent(next_state)[0]
            # Create targets
            for i in range(self.batch_size):
                if done[i]:
                    q_values_t[i][action[i]] = reward[i]
                else:
                    q_values_t[i][action[i]] = reward[i] + \
                                               self.gamma * target_agent_q_values_s_[i][q_values_t1[i]]

        predictions, _ = agent(state)

        # Task Policy head training
        # noinspection PyCallingNonCallable
        loss = self.criterion(predictions, q_values_t).mean()
        loss.backward()
        self.optimizer.step()

        return loss


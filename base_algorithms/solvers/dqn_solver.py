import random
import sys, os
from os.path import join
import json
import numpy as np

import torch

from .base_solver import BaseSolver

CUR = os.path.abspath(os.path.dirname(__file__))


class DQNSolver(BaseSolver):
    def __init__(self, cfg, agent=None, dataloader=None, **kwargs):
        super(DQNSolver, self).__init__(cfg, agent, dataloader, **kwargs)

        self.ddqn_prob = 0
        if agent is None:
            self.target_agent = self.create_agent(cfg.MODEL.TYPE,
                                                cfg.MODEL.ARCH,
                                                cfg.MODEL.MISC,
                                                cfg.MODEL.INIT)

    def train(self):
        itx = 0
        best_score = 0.
        json_path = join(CUR, self.save_path)
        log_path = json_path[:-4] + 'txt'

        with open(log_path, 'w') as loss_file:
            loss_file.write('Metrics:' + '\n')

        with open(json_path, 'w') as json_file:
            json.dump(self.cfg, json_file, sort_keys=True,
                      indent=4, separators=(',', ': '))

        for episode in range(self.max_episodes):
            loss = 0.
            if episode >= self.wait_episodes:
                print('Training at episode: {}'.format(episode + 1))
            else:
                print('Burn in period. Not training yet')
            s = self.env.reset()
            for t in range(self.max_steps_per_episode):
                # Epsilon greedy with eps = 1/(itx+1)
                action, eps = self.epsilon_greedy(itx, s)
                s_, r, d, _ = self.env.step(action)
                if d:
                    r = -100.
                self.remember(self.preprocess(s),
                              action,
                              r,
                              self.preprocess(s_),
                              d)
                s = s_
                if episode >= self.wait_episodes:
                    if np.random.uniform(0, 1) > self.ddqn_prob:
                        loss += self.train_step(self.agent, self.target_agent)
                    else:
                        loss += self.train_step(self.target_agent, self.agent)

                itx += 1
                if d:
                    loss = float("{:.2f}".format(loss / (t + 1)))
                    episode_str = 'Episode length: {} with final eps: {}, and loss {}'.format(t + 1, eps, loss)
                    print(episode_str)
                    mu, std, bot_mu = self.validate_agent()
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
                self.save_agent(self.weights_path, episode)

        print('Finished training!')

    def train_step(self, agent, target_agent):
        if len(self.memory) < self.batch_size:
            return 0.
        minibatch = random.sample(self.memory, self.batch_size)
        state = tf.stack([minibatch[i][0][0] for i in range(self.batch_size)])
        action = [minibatch[i][1] for i in range(self.batch_size)]
        reward = [minibatch[i][2] for i in range(self.batch_size)]
        next_state = tf.stack([minibatch[i][3][0] for i in range(self.batch_size)])
        done = [minibatch[i][4] for i in range(self.batch_size)]

        q_values_t = agent.predict(state)
        q_values_t1 = tf.math.argmax(agent.predict(next_state), axis=-1)

        for i in range(self.batch_size):
            if done[i]:
                q_values_t[i][action[i]] = reward[i]
            else:
                q_values_t[i][action[i]] = reward[i] + \
                                           self.gamma * target_agent.predict(next_state)[i][q_values_t1[i]]

        with tf.GradientTape() as tape:
            predictions = agent(state, training=True)
            loss = self.criterion(q_values_t, predictions)
            grads = tape.gradient(loss, agent.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, agent.trainable_variables))

            return loss
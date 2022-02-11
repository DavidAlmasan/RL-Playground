"""
Implements solver for ActorCritic models
TODO: port to pytorch
"""
import random
import os
from os.path import join
import json
import numpy as np
from itertools import accumulate

import torch

from base_algorithms.solvers.base_solver import BaseSolver

CUR = os.path.abspath(os.path.dirname(__file__))


class A2CSolver(BaseSolver):
    """
    Child Solver class for ActorCritic models
    """
    def __init__(self, cfg):
        super(A2CSolver, self).__init__(cfg)

    def remember(self, state, action, value, reward, next_state, done):
        self.memory.append((state, action, value, reward, next_state, done))

    def store_episode(self, idx):
        state = self.env.reset()
        done = False
        t = 0
        states, actions, values, rewards, next_states, dones = [], [], [], [], [], []

        while not done and t < self.max_steps_per_episode:

            state = self.preprocess(state)
            self.agent.eval()
            action_space, value_space = self.agent(state)
            action = tf.math.argmax(action_space[0]).numpy()

            # Use epsilon greedy
            # action, eps = self.epsilon_greedy(idx, state)
            value = value_space[0][0].numpy()
            next_state, reward, done, _ = self.env.step(action)
            if done:
                reward = -100
            states.append(state[0])
            actions.append(action)
            values.append(value)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            state = next_state
            idx += 1
            t += 1

        rewards = self.compute_cumulative_rewards(rewards)
        for s, a, v, r, s_, d in zip(states,
                                  actions,
                                  values,
                                  rewards,
                                  next_states,
                                  dones):
            self.remember(s, a, v, r, s_, d)
        episode = [tf.stack(val) for val in [states, actions, values, rewards, next_states, dones]]
        return idx, episode

    def compute_cumulative_rewards(self, rewards, normalise=False):
        rewards = rewards[::-1]
        rewards = list(accumulate(rewards, lambda x, y: x * self.gamma + y))[::-1]
        if normalise:
            rewards = list(np.asarray(rewards) / max(rewards))

        return rewards

    def epsilon_greedy(self, s):
        """
        Epsilon-greedy strategy for action selection at time t, given state s
        """
        self.agent.eval()
        eps = max(0.1, self.eps_reduction_factor ** self.train_iter)
        eps = float("{:.2f}".format(eps))
        action_space, value = self.agent(s)
        if eps <= np.random.uniform(0, 0.9999):
            # Perform explore action
            action = self.env.action_space.sample()
        else:
            # Perform greedy action
            action = torch.argmax(action_space, dim=-1).numpy()
        self.agent.train()
        return action, value

    def train(self):
        self.logger.info(f'Training Agent using A2C Method')
        best_score = 0.
        loss = 0
        json_path = join(self.save_path, 'info.json')
        log_path = json_path[:-4] + 'txt'

        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.cfg.to_json(), json_file, ensure_ascii=False, indent=4)
        with open(log_path, 'w') as loss_file:
            loss_file.write('Metrics:' + '\n')
        s = self.env.reset()

        # TODO: Finish this
        while True:
            if self.stopping_criterion():
                break
            s = self.preprocess(s)
            action, value = self.epsilon_greedy(s)
            s_, r, d, info = self.env.step(action)
            self.remember(s,
                          action,
                          value,
                          r,
                          s_,
                          d)
            # Update to next states
            s = s_
            # Train step
            loss = self.train_step(self.agent, self.target_agent)

            self.logger.info(f'Loss: {loss}; Iter: {self.train_iter}')
            self.train_iter += 1

            if self.train_iter % self.validation_iters == 0:
                self.validate_agent()


        ### Old
        best_score = 0.
        json_path = join(CUR, self.save_path)


        with open(json_path, 'w') as json_file:
            json.dump(self.cfg, json_file, sort_keys=True,
                      indent=4, separators=(',', ': '))

        for episode in range(self.max_episodes):
            itx, ep = self.store_episode(itx)
            loss = 0.
            if episode >= self.wait_episodes:
                if not async_solver:
                    print('Training at episode: {}'.format(episode + 1))

                # Train on episode
                ep_loss, ep_grads = self.train_on_episode(self.agent, ep)
                loss += ep_loss
                if episode == self.wait_episodes:
                    grads = ep_grads
                else:
                    # noinspection PyUnboundLocalVariable
                    grads += ep_grads

                # Train on experience replay
                # loss += self.train_on_replay(self.agent)

                if not async_solver:
                    mu, std, bot_mu = self.validate_agent(multihead=True)
                    log_str = 'mean: {}, std: {}, bottom_mean: {}, loss: {}'.format(mu, std, bot_mu, loss)
                    if mu > best_score:
                        best_score = mu
                        self.save_agent(self.weights_path, 'best')
                    with open(log_path, 'a') as loss_file:
                        loss_file.write(log_str + '\n')
            else:
                print('Burn in period. Not training yet')
            if not async_solver:
                if episode in self.log_episodes:
                    self.save_agent(self.weights_path, episode)
        if not async_solver:
            print('Finished training')
        else:
            return grads

    def train_on_episode(self, agent, episode):
        state = episode[0]
        action = episode[1]
        reward = episode[3]

        # Refactor actions in gather format
        action_idx = tf.expand_dims(tf.range(0, len(action)), axis=1)
        action_indeces = tf.concat([action_idx, tf.expand_dims(action, axis=1)], axis=-1)

        loss, grads = self.train_step(agent, state, action_indeces, reward)
        return loss, grads

    def train_on_replay(self, agent):
        if len(self.memory) < self.batch_size:
            return 0.
        minibatch = random.sample(self.memory, self.batch_size)
        state = tf.stack([minibatch[i][0] for i in range(self.batch_size)])
        action = [minibatch[i][1] for i in range(self.batch_size)]
        reward = tf.stack([minibatch[i][3] for i in range(self.batch_size)])

        # Refactor actions in gather format
        action_idx = tf.expand_dims(tf.range(0, len(action)), axis=1)
        action_indeces = tf.concat([action_idx, tf.expand_dims(action, axis=1)], axis=-1)

        loss, grads = self.train_step(agent, state, action_indeces, reward)
        return loss, grads

    def train_step(self, agent, state):
        with tf.GradientTape() as tape:
            q_values_t, state_values_t = agent(state)
            q_values_probs = tf.nn.softmax(q_values_t)
            q_probs = tf.gather_nd(q_values_probs, action_indeces)
            log_probs = tf.math.log(q_probs)
            advantage = reward - state_values_t

            # Actor loss
            actor_loss = -tf.reduce_sum(log_probs * advantage)

            # Critic loss
            critic_loss = self.criterion(reward, state_values_t)

            loss = actor_loss + critic_loss

            # Gradients
            grads = tape.gradient(loss, agent.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, agent.trainable_variables))

            return loss, grads


import sys, os
import numpy as np

from solvers.dqn_solver import DQNSolver


class DDQNSolver(DQNSolver):
    def __init__(self, cfg):
        super(DDQNSolver, self).__init__(cfg)
        self.ddqn_prob = 0.5

    def epsilon_greedy(self, t, s):
        eps = self.eps_red_factor ** t
        eps = max(0.01, eps)
        eps = float("{:.2f}".format(eps))
        s = self.preprocess(s)
        action_space = np.squeeze(self.agent(s).numpy() + self.target_agent(s).numpy())
        if eps <= np.random.uniform(0, 0.9999):
            # Perform explore action
            action = self.env.action_space.sample()
        else:
            # Perform greedy action
            action = np.argmax(action_space)
        return action, eps



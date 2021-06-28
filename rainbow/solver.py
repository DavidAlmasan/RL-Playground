import numpy as np
import torch
from PIL import Image

from base_algorithms.solvers.base_solver import BaseSolver


class RainbowSolver(BaseSolver):
    def __init__(self, cfg):
        super(RainbowSolver, self).__init__(cfg)


    def preprocess(self, states):
        return torch.unsqueeze(torch.stack([torch.squeeze(self.transforms(Image.fromarray(s))) for s in states]), dim=0)

    def select_action(self, idx, s, type='epsilon_greedy'):
        if type == "epsilon_greedy":
            return self.epsilon_greedy(idx, s)
        else:
            raise NotImplementedError('Only epislon greedy selection scheme supported')

    def populate_memory(self, num_samples):
        sample_idx = 0
        while True:
            start = self.env.reset()
            s = [np.zeros_like(start)] * (self.cfg.AGENT.HYPERPARAMS.NUM_EPISODES_PER_SAMPLE - 1) + [start]
            for t in range(self.max_steps_per_episode):
                preprocessed_s = self.preprocess(s)
                # Epsilon greedy with eps = 1/(itx+1)
                action, eps = self.select_action(sample_idx, preprocessed_s)
                s_, r, d, _ = self.env.step(action)
                s_ = s[-(self.cfg.AGENT.HYPERPARAMS.NUM_EPISODES_PER_SAMPLE - 1):] + [s_]
                if d:
                    r = -self.max_steps_per_episode
                self.remember(preprocessed_s,
                              action,
                              r,
                              self.preprocess(s_),
                              d)
                s = s_
                sample_idx += 1
                if sample_idx == num_samples:
                    return


    def solve(self):
        """
        1. Play some episodes using the untrained agent and store them in the memory
            1.1 Use <Prioritied replay>
        2. Start training using samples from the memory
        """

        self.populate_memory(self.cfg.TRAIN.WAIT_SAMPLES)
        print(self.memory[0][0].shape)
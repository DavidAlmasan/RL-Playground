import numpy as np
import torch
from PIL import Image

from base_algorithms.solvers.base_solver import BaseSolver
from rainbow.replay_buffer import ReplayBuffer



class RainbowSolver(BaseSolver):
    def __init__(self, cfg):
        super(RainbowSolver, self).__init__(cfg)
        self.memory = ReplayBuffer(self.cfg.RAINBOW.REPLAY_BUFFER)
        
    def epsilon_greedy(self, t, s, ):
        eps = max(0.1, self.eps_reduction_factor ** t)
        eps = float("{:.2f}".format(eps))
        q_values, probs = self.agent(s)
        if eps <= np.random.uniform(0, 0.9999):
            # Perform explore action
            action = self.env.action_space.sample()
        else:
            # Perform greedy action
            action = torch.argmax(probs, dim=-1)
        return action, eps

    def preprocess(self, states):
        return torch.unsqueeze(torch.stack([torch.squeeze(self.transforms(Image.fromarray(s))) for s in states]), dim=0)

    def select_action(self, idx, s, type='epsilon_greedy'):
        if type == "epsilon_greedy":
            return self.epsilon_greedy(idx, s)
        else:
            raise NotImplementedError('Only epislon greedy selection scheme supported')

    def solve(self):
        """
        1. Play some episodes using the untrained agent and store them in the memory
            1.1 Use <Prioritied replay>
        2. Start training using samples from the memory every REPLAY_PERIOD steps
        """
        sample_idx = 0
        episodes_played = 0
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
                              d,
                              sample_idx)
                s = s_
                sample_idx += 1

                if sample_idx % self.cfg.TRAIN.REPLAY_PERIOD == 0 and sample_idx >= self.cfg.TRAIN.BATCH_SIZE:
                    # Sample form the Replay memory and train agent
                    batch_samples = self.memory.sample_transition(sample_idx, self.cfg.TRAIN.BATCH_SIZE)







                

    def remember(self, state, action, reward, next_state, done, sample_idx):
        transition = (state, action, reward, next_state, done)
        self.memory.populate(transition, sample_idx)

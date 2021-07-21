from random import sample
from time import sleep
import sys
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
        eps = max(0.05, self.eps_reduction_factor ** t)
        eps = float("{:.2f}".format(eps))
        self.agent.eval()
        _, probs = self.agent(s)
        self.agent.train()
        if eps <= np.random.uniform(0, 0.9999):
            # Perform explore action
            action = torch.tensor(self.env.action_space.sample())
        else:
            # Perform greedy action
            action = torch.argmax(probs, dim=-1).squeeze()
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
        train_idx = 0
        episodes_played = 0
        while True:
            start = self.env.reset()
            s = [np.zeros_like(start)] * (self.cfg.AGENT.HYPERPARAMS.NUM_FRAMES_PER_SAMPLE - 1) + [start]
            for t in range(self.max_steps_per_episode):
                preprocessed_s = self.preprocess(s)
                # Epsilon greedy with eps = 1/(itx+1)
                action, eps = self.select_action(sample_idx, preprocessed_s)
                s_, r, d, _ = self.env.step(action)
                s_ = s[-(self.cfg.AGENT.HYPERPARAMS.NUM_FRAMES_PER_SAMPLE - 1):] + [s_]
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

                if sample_idx % self.cfg.TRAIN.REPLAY_PERIOD == 0 and sample_idx >= self.batch_size:
                    # Sample form the Replay memory and train agent
                    batch_samples = self.memory.sample_transition(sample_idx, self.batch_size)
                    # batch_s = torch.stack([sample.transition[0] for sample in batch_samples])
                    # batch_a = torch.stack([sample.transition[1] for sample in batch_samples])
                    # batch_r = torch.stack([torch.tensor(sample.transition[2]) for sample in batch_samples])
                    # batch_s_ = torch.stack([sample.transition[3] for sample in batch_samples])

                    # Compute TD Error and back prop
                    td_errors, idx_list, rewards = self.train_step(train_idx, batch_samples)
                    print(f'Iteration: {sample_idx}, TD_avg: {torch.stack(td_errors).mean()}, Reward_avg: {rewards:.4f}')

                    # Update priorities in the ReplayBuffer 
                    self.memory.update_priorities(idx_list, td_errors)

                    train_idx += 1
                
                if sample_idx % self.cfg.TRAIN.EVAL_PERIOD == 0:
                    self.validate_agent(games=10, num_steps=self.max_steps_per_valepisode, render=False)

                if sample_idx % self.cfg.TRAIN.TOT_TRAIN_STEPS == 0:
                    self.validate_agent(games=1, num_steps=self.max_steps_per_episode, render=True)
                    return

    @torch.no_grad()
    def construct_gt(self, batch_s, batch_a, batch_r, batch_s_, batch_done):
        self.agent.eval()
        self.target_agent.eval()
        Q_values_gt, _ = self.agent(batch_s)
        Q_values_s_, _ = self.agent(batch_s_)
        targetagent_Q_values_s_, _ = self.agent(batch_s_)
        max_actions_s_ = torch.argmax(Q_values_s_, dim=-1)

        for batch in range(self.batch_size):
            try:
                if batch_done[batch]:
                    Q_values_gt[batch, batch_a[batch]] = batch_r[batch]
                else:
                    Q_values_gt[batch, batch_a[batch]] = batch_r[batch] + \
                                        self.gamma * targetagent_Q_values_s_[batch, max_actions_s_[batch]]  

            except:
                print(self.batch_size)
                print(batch_done.shape)
                sys.exit()

        self.agent.train()

        return Q_values_gt

    def train_step(self, train_idx, batch_samples):
        batch_s = torch.stack([sample.transition[0] for sample in batch_samples]).squeeze()
        batch_a = torch.stack([sample.transition[1] for sample in batch_samples])
        batch_r = torch.stack([torch.tensor(sample.transition[2]) for sample in batch_samples])
        batch_s_ = torch.stack([sample.transition[3] for sample in batch_samples]).squeeze()
        batch_done = torch.stack([torch.tensor(sample.transition[4]) for sample in batch_samples])

        sampleidxs = [sample.index for sample in batch_samples]


        if train_idx % self.cfg.RAINBOW.TARGET_NET_COPY_TRAINSTEPS == 0:
            self.agent.eval()
            self.target_agent.eval()
            self.target_agent.load_state_dict(self.agent.state_dict())
            self.agent.train()

        # Construct Q value gt 
        gt = self.construct_gt(batch_s, batch_a, batch_r, batch_s_, batch_done)

        # Q-learning step
        predictions, _ = self.agent(batch_s)
        loss = self.criterion(gt, predictions)
        td_errors = []
        for batch in range(self.batch_size):
            td_error = torch.square(loss[batch, batch_a[batch]]).detach()
            td_errors.append(td_error)

        # Backprop
        loss.sum(dim=-1).mean(dim=0).backward()

        return td_errors, sampleidxs, batch_r.mean()

    def remember(self, state, action, reward, next_state, done, sample_idx):
        transition = (state, action, reward, next_state, done)
        self.memory.populate(transition, sample_idx)

    def play(self, num_steps, render):
        """
        :param environment: env
        :param policy: agent that plays in given env
        :param num_steps: steps before episode termination
        :param render: whether to render the env
        :return: length of the episode
        """
        tot_r = 0
        start = self.val_env.reset()
        s = [np.copy(start)] * (self.cfg.AGENT.HYPERPARAMS.NUM_FRAMES_PER_SAMPLE - 1) + [start]
        for t in range(num_steps):
            if render:
                self.val_env.render()
                
            preprocessed_s = self.preprocess(s)
            # Epsilon greedy with eps = 1/(itx+1)
            _, probs = self.agent(preprocessed_s)
            action = torch.argmax(probs, dim=-1).squeeze() if t > 0 else 1
            s_, r, d, _ = self.val_env.step(action)
            tot_r += r
            s_ = s[-(self.cfg.AGENT.HYPERPARAMS.NUM_FRAMES_PER_SAMPLE - 1):] + [s_]
            if d:
                if render:
                    print('Replay Finished at time step: {}'.format(t + 1))
            s = s_
            
        if render:
            print('Achieved max steps!!')
        return tot_r

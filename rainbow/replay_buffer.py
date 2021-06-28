from collections import deque, namedtuple
from itertools import islice
import random
from numpy.core.fromnumeric import _all_dispatcher

import torch
import numpy as np


class ReplayBuffer():
    def __init__(self, cfg):
        self.cfg = cfg

        self.memory_len = self.cfg.MEMORY_LEN
        self.alpha = self.cfg.ALPHA
        self.beta = self.cfg.BETA
        self.max_p = 1
        self.create_data = namedtuple("Data", field_names=['transition', 'priority', 'probability', 'weight', 'index'])
        # Create the memory dict
        self.memory = {key: self.create_data((0,0,0,0,0), 0, 0, 0, 1) for key in range(self.memory_len)}
        # Sum_i (p_i^alpha)
        self.priority_normalising_sum = 0
        # Sum_i (w_i)
        self.max_w = 0
        self.max_p = 1

        self.N = 10 # TODO see what the N is in https://arxiv.org/pdf/1511.05952.pdf, page 5
     
    def sample_transition(self, idx, batch_size):
        if idx >= self.memory_len:
            idx = self.memory_len
        num_bins, remainder = divmod(idx, batch_size)
        splits = [batch_size] * num_bins
        if remainder != 0:
            splits += [remainder]
        idx_iter = iter(list(range(idx)))
        splits = [list(islice(idx_iter, elem)) for elem in splits]

        batch_samples = []
        for idx_batch in splits:
            sample_probs = [self.memory[i].probability for i in idx_batch]
            sample_idx = random.choices(idx_batch, weights=sample_probs)
            batch_samples.append(self.memory[sample_idx])
        return batch_samples
            
    def update_transitions(self, samples_batch, td_errors):
        batch_size = len(samples_batch)

        for batch in range(batch_size):
            sample = samples_batch[batch]

            idx = sample.index
            temp = self.memory[idx]
            self.memory[idx].priority = td_errors[batch]

            if self.memory[idx].priority > self.max_p:
                self.max_p = self.memory[idx].priority
            
            self.priority_normalising_sum -= temp.priority ** self.alpha
            self.priority_normalising_sum += self.memory[idx].priority ** self.alpha

            self.memory[idx].probability = self.memory[idx].priority ** self.alpha / self.priority_normalising_sum
            self.memory[idx].weight = (self.N * self.memory[idx].probability) ** self.beta
            if self.memory[idx].weight > self.max_w:
                self.max_w = self.memory[idx].weight

            
            

    def populate(self, transition, idx, compute_weights):
        q, idx = divmod(idx, self.memory_len)

        if q > 0:
            prev_entry = self.memory[idx]
            # Erase info from last entry 
            self.memory[idx].probability = 0
            
            # Adjust priority_normalising_sum
            self.priority_normalising_sum -= prev_entry.priority ** self.alpha
            if prev_entry.priority == self.max_p:
                self.memory[idx].priority = 0
                self.max_p = max(self.memory.values(), key=lambda x: x.priority)
            
            if compute_weights:
                if prev_entry.weight == self.max_w:
                    self.memory[idx].weight = 0
                    self.max_w = max(self.memory.values(), key=lambda x: x.weight)
            
        priority = self.max_p
        self.priority_normalising_sum += priority ** self.alpha
        probability = priority ** self.alpha / self.priority_normalising_sum
        weight = self.max_w
        self.memory[idx].transition = transition
        self.memory[idx].priority = priority
        self.memory[idx].probability = probability
        self.memory[idx].weight = weight

        






                

    


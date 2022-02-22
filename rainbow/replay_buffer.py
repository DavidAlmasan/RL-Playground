from collections import deque, namedtuple
from dataclasses import dataclass
from itertools import islice
import random
from numpy.core.fromnumeric import _all_dispatcher, _transpose_dispatcher

import torch
import numpy as np


@dataclass
class Data:
    transition: tuple
    priority: int
    probability: float 
    weight: float 
    index: int 

class ReplayBuffer:
    """
    Replay Buffer as in  https://arxiv.org/pdf/1511.05952.pdf
    Changes to bring: Maybe use binary heap as described in the paper instead of dict 
    and namedtuple?

    """
    def __init__(self, cfg):
        self.cfg = cfg

        self.memory_len = int(self.cfg.MEMORY_LEN)  # N
        self.alpha = self.cfg.ALPHA
        self.beta = self.cfg.BETA
        self.max_p = 1
        self.create_data = namedtuple("Data", field_names=['transition', 'priority', 'probability', 'weight', 'index'])
        # Create the memory dict
        self.memory = {key: Data((0, 0, 0, 0, 0), 0, 0.0, 0.0, 1) for key in range(self.memory_len)}
        # Sum_i (p_i^alpha)
        self.priority_normalising_sum = 0
        # Sum_i (w_i)
        self.max_w = 0
        self.max_p = 1

     
    def sample_transition(self, idx, batch_size):
        # TODO: Move to binary heap instead of sampling according to the probabilities.
        # TODO: Right now we are not spliting the PROBABILITY range into equal parts but rather the MEMORY
        # then we sample according to those probabilities. i guess this works but it's weird

        if idx >= self.memory_len:
            idx = self.memory_len
        # Update all probabilities and IS weights
        self.update_transitions()

        binsize, remainder = divmod(idx, batch_size)
        if remainder > 0:
            binsize, remainder = divmod(idx + (binsize - remainder), batch_size)
        # print(binsize, batch_size, remainder, idx)
        # splits = [batch_size] * num_bins
        # if remainder != 0:
        #     splits += [remainder]
        # idx_iter = iter(list(range(idx)))
        # splits = [list(islice(idx_iter, elem)) for elem in splits]
        indices = list(range(idx))
        splits = [indices[i:i+binsize] for i in range(0, idx, binsize)]
        # print('replay_buffer#l64: Length of splits is {}, should be {}. Extras: '.format(len(splits), batch_size), binsize, remainder, idx)
        batch_samples = []
        for idx_batch in splits:
            sample_probs = [self.memory[i].probability for i in idx_batch]
            sample_idx = random.choices(idx_batch, weights=sample_probs)[0]  # For some reason this is returned as a list ??
            while sample_idx >= idx:
                sample_idx = random.choices(idx_batch, weights=sample_probs)[0]
            self.memory[sample_idx].weight = torch.sigmoid(
                            torch.tensor((self.memory_len * self.memory[sample_idx].probability) ** self.beta))
            batch_samples.append(self.memory[sample_idx])

        # Adjust weights to be 0-1
        return batch_samples
            
    def update_transitions(self):
        """Updates all the probabilies and the IS weights needed for backprop"""
        # TODO: THIS IS NOT OPTIMISED. i think
        # self.max_w = 0
        for key in range(self.memory_len):
            self.memory[key].probability = self.memory[key].priority ** self.alpha / self.priority_normalising_sum
            # self.memory[key].weight = (self.memory_len * self.memory[key].probability) ** self.beta
            # if self.memory[key].weight > self.max_w:
            #     self.max_w = self.memory[key].weight
        



    def update_transitions_old(self, samples_batch, td_errors):
        """
        Old
        """
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
            self.memory[idx].weight = (self.memory_len * self.memory[idx].probability) ** self.beta
            if self.memory[idx].weight > self.max_w:
                self.max_w = self.memory[idx].weight

            
    def update_priorities(self, idx_list, new_priorities):
        for idx, new_p in zip(idx_list, new_priorities):
            prev_p = self.memory[idx].priority

            # Update normalising factor
            self.priority_normalising_sum -= prev_p ** self.alpha
            self.priority_normalising_sum -= new_p ** self.alpha

            self.memory[idx].priority = new_p

            if new_p >= self.max_p:
                self.max_p = new_p

    def populate(self, transition, idx):
        q, idx = divmod(idx, self.memory_len)

        if q > 0:
            prev_entry = self.memory[idx]
            # Erase info from last entry 
            self.memory[idx].probability = torch.tensor(0)
            
            # Adjust priority_normalising_sum
            self.priority_normalising_sum -= prev_entry.priority ** self.alpha
            if prev_entry.priority == self.max_p:
                self.memory[idx].priority = torch.tensor(0)
                self.max_p = max(self.memory.values(), key=lambda x: x.priority)
            
            # if compute_weights:
            #     if prev_entry.weight == self.max_w:
            #         self.memory[idx].weight = 0
            #         self.max_w = max(self.memory.values(), key=lambda x: x.weight)
            
        priority = self.max_p
        self.priority_normalising_sum += priority ** self.alpha
        probability = priority ** self.alpha / self.priority_normalising_sum
        weight = self.max_w
        # print(self.memory[idx])
        # print(type(priority), priority)
        # print(type(probability), probability)
        # print(type(weight), weight)
        # import sys
        # sys.exit()
        self.memory[idx].transition = transition
        self.memory[idx].priority = priority
        self.memory[idx].probability = probability
        self.memory[idx].weight = weight

        






                

    


"""
Implements various util functions
"""
import sys, os
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import torch




def play_tf(environment, policy=None, num_steps=1000, render=True, multihead=False):
    """
    :param environment: env
    :param policy: agent that plays in given env
    :param num_steps: steps before episode termination
    :param render: whether to render the env
    :return: length of the episode
    """
    s = environment.reset()

    if policy is None:
        for step in range(num_steps):
            if render:
                environment.render()
            s, r, d, _ = environment.step(environment.action_space.sample())
        return

    for step in range(num_steps):
        if render:
            environment.render()
        s = np.expand_dims(np.asarray(s), axis=0)
        if multihead:
            actions = np.squeeze(policy(s, training=False)[0].numpy())
        else:
            actions = np.squeeze(policy(s, training=False).numpy())
        action = np.argmax(actions)
        s, r, d, _ = environment.step(action)
        if d:
            if render:
                print('Replay Finished at time step: {}'.format(step + 1))
            return step + 1
    if render:
        print('Achieved max steps!!')
    return num_steps

def bottom_n_percent(l, n):
    n = int(len(l) * n / 100.)
    if n == 0:
        n = 1
    return sum(sorted(l)[:n]) / n

def plot_metrics(metrics_file):
    with open(metrics_file, 'r') as file_:
        lines = file_.readlines()[1:]
    mu, bot, loss = [], [], []
    for line in lines:
        line = line.rstrip().split(',')
        mu.append(float(line[0].split(':')[-1]))
        bot.append(float(line[2].split(':')[-1]))
        loss.append(float(line[3].split(':')[-1]))
    x = list(range(len(mu)))

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(x, loss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Steps/ep', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, mu, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # color = 'tab:green'
    # ax3.set_ylabel('BotSteps/ep', color=color)  # we already handled the x-label with ax1
    # ax3.plot(x, bot, color=color)
    # ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

# This module is used to _try_ to implement Rainbow as in :
[Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298.pdf)

# TODO
* Set up environment
* Set up an agent (a model)

# Elements of Rainbow:
1. DQN (see ../base_algorithms)
2. Double Q learning (see ../base_algorithms)
3. Dueling Networks (see ../base_algorithms)
3. Prioritied replay
![][logo]
[logo]: https://github.com/DavidAlmasan/RL-Playground/tree/master/rainbow/prioritised_replay.png "Equation stolen from Rainbow paper"
4. Multi step return: Use truncated _n_-step return 
5. [Distributional RL](https://arxiv.org/pdf/1707.06887.pdf)
6. [Noisy Networks for Exploration](https://arxiv.org/pdf/1706.10295.pdf)


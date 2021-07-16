
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

d = {key: Data((0, 0, 0, 0, 0), 100, 0.0, 0.0, 1) for key in range(10)}
print(d)
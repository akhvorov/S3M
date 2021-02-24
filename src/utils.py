import numpy as np

import random
import torch

random_seed = 5


def set_seed(seed: int = random_seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

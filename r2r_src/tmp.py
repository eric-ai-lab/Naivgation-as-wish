import torch
import numpy as np
import numpy.random as random
raw_noise = np.random.uniform(low = -1, high = 1, size = 512)
attack_pos = np.random.binomial(n = 1, p = 1, size = 512)
attack_noise = torch.from_numpy(raw_noise * attack_pos)

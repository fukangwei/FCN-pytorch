import numpy as np


def onehot(data, n):
    buf = np.zeros(data.shape + (n,))
    all_one = np.ones(data.shape)
    buf[:, :, 0] = data
    buf[:, :, 1] = all_one - data
    return buf

import copy
import torch.nn as nn
import torch


# Helper for constructing the one-hot vectors.
def construct_maps(keys):
    d = dict()
    keys = list(set(keys))
    for k in keys:
        if k not in d:
            d[k] = len(list(d.keys()))
    return d


ks_map = construct_maps(keys=(3, 5, 7))
ex_map = construct_maps(keys=(3, 4, 6))
dp_map = construct_maps(keys=(2, 3, 4))


def spec2feats(ks_list, ex_list, d_list, r):
    # This function converts a network config to a feature vector (128-D).
    start = 0
    end = 4
    for d in d_list:
        for j in range(start + d, end):
            ks_list[j] = 0
            ex_list[j] = 0
        start += 4
        end += 4

    # convert to onehot
    ks_onehot = [0 for _ in range(60)]
    ex_onehot = [0 for _ in range(60)]
    r_onehot = [0 for _ in range(8)]

    for i in range(20):
        start = i * 3
        if ks_list[i] != 0:
            ks_onehot[start + ks_map[ks_list[i]]] = 1
        if ex_list[i] != 0:
            ex_onehot[start + ex_map[ex_list[i]]] = 1

    r_onehot[(r - 112) // 16] = 1
    return torch.Tensor(ks_onehot + ex_onehot + r_onehot)


def latency_encoding(child_arch):
    ks_list = copy.deepcopy(child_arch['ks'])
    ex_list = copy.deepcopy(child_arch['e'])
    d_list = copy.deepcopy(child_arch['d'])
    r = copy.deepcopy(child_arch['r'])[0]
    feats = spec2feats(ks_list, ex_list, d_list, r).reshape(1, -1)
    return feats


if __name__ == '__main__':
    child_arch = {'wid': None, 'ks': [3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5], 'e': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4], 'd': [3, 2, 2, 3, 3], 'r': [176]}
    one_hot = latency_encoding(child_arch)
    print(one_hot)


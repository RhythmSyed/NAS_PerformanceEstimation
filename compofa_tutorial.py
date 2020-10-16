import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
import time
import random
import math
import copy

from matplotlib import pyplot as plt

import sys
sys.path.append("..")
from ofa.model_zoo import ofa_net
from ofa.utils import download_url

from accuracy_predictor import AccuracyPredictor
from flops_table import FLOPsTable
from latency_table import LatencyTable
from evolution_finder import EvolutionFinder
from imagenet_eval_helper import evaluate_ofa_subnet, evaluate_ofa_specialized

# set random seed
random_seed = 10291284
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)


cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(random_seed)
    print('Using GPU.')
else:
    print('Using CPU.')

ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)
# ofa_network_2 = ofa_net('ofa', pretrained=True)
print('The OFA Network is ready.')

data_loader = None


# accuracy predictor
accuracy_predictor = AccuracyPredictor(
    pretrained=True,
    device='cuda:0' if cuda_available else 'cpu'
)

print('The accuracy predictor is ready!')
print(accuracy_predictor.model)


target_hardware = 'note10'
latency_table = LatencyTable(device=target_hardware, use_latency_table=False)
print('The Latency lookup table on %s is ready!' % target_hardware)



""" Hyper-parameters for the evolutionary search process
    You can modify these hyper-parameters to see how they influence the final ImageNet accuracy of the search sub-net.
"""
latency_constraint = 25  # ms, suggested range [15, 33] ms
P = 100  # The size of population in each generation
N = 500  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation
params = {
    'constraint_type': target_hardware, # Let's do FLOPs-constrained search
    'efficiency_constraint': latency_constraint,
    'mutate_prob': 0.1, # The probability of mutation in evolutionary search
    'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
    'efficiency_predictor': latency_table, # To use a predefined efficiency predictor.
    'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
    'population_size': P,
    'max_time_budget': N,
    'parent_ratio': r,
    'arch' : 'compofa', ## change
}

# build the evolution finder
finder = EvolutionFinder(**params)

"""
# start searching
result_lis = []
st = time.time()
best_valids, best_info = finder.run_evolution_search()
result_lis.append(best_info)
ed = time.time()
print('Found best architecture on %s with latency <= %.2f ms in %.2f seconds! '
      'It achieves %.2f%s predicted accuracy with %.2f ms latency on %s.' %
      (target_hardware, latency_constraint, ed-st, best_info[0] * 100, '%', best_info[-1], target_hardware))
"""


result_lis = []
for latency in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:
    st = time.time()
    finder.set_efficiency_constraint(latency)
    best_valids, best_info = finder.run_evolution_search()
    ed = time.time()
    result_lis.append(best_info)
print("Done!")

"""
# visualize the architecture of the searched sub-net
_, net_config, latency = best_info
ofa_network.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
print('Architecture of the searched sub-net:')
print(ofa_network.module_str)
"""
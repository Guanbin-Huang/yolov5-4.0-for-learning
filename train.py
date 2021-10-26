"""
This is train.py. So we usually need to import model and dataset and some other utils
"""

import models
import dataset
import torch
import torch.nn as nn
import nn_utils
import sys_utils

'''
Let's write the main part
'''

def train():
    pass

if __name__ == "__main__":
    # some hyper parameters
    batch_size = 2
    device = 0

    nn_utils.setup_seed(3)
    train()
    torch.cuda.empty_cache() # Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible in nvidia-smi.
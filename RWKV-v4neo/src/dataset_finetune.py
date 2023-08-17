########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
# from .binidx import MMapIndexedDataset
from .dataloader import dataloader
import random
import copy

train_steps = 500

class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.vocab_size = args.vocab_size
        self.data = dataloader(args.data_file)
        self.pool = self.limit_sample(args.epoch_steps)
        self.data_size = len(self.data)
        rank_zero_info(f"Data has {self.data_size} items.")
        self.item = []
 
    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def limit_sample(self,n):
        if len(self.data) <= n:
            res = random.sample(self.data, len(self.data))
        else:
            res = random.sample(self.data, n)
        return res

    def __getitem__(self, idx):
        args = self.args
        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        if len(self.item) == 0:
            if len(self.pool) == 0:
                self.pool = self.limit_sample(args.epoch_steps)
            self.item = self.pool[0]
            self.pool = self.pool[1:]
        step  = self.item[:req_len]
        step_len =  len(step)
        if len(self.item) > req_len:
            half = int(ctx_len / 2)
            self.item = self.item[half:]
        else:
            self.item = self.item[req_len:]
        dix = [0 for x in range(req_len)]
        dix[:step_len] = step
        mask = [int(x!=0) for x in dix]
        mask = mask[:-1]
        start = dix[:-1]
        l = random.randrange(0,ctx_len)
        n = random.randrange(0,65536)
        start[l] = n
        # x = torch.tensor(dix[:-1], dtype=torch.long)
        x = torch.tensor(start, dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        z = torch.tensor(mask, dtype=torch.float32)
        return x, y, z

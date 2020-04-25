"""
Created on Mon Feb 24 2020

@author: fanghenshao
"""

import torch
import numpy as np
import random

import os
import json


# -------- fix random seed 
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# -------- SVAE OPTIONS --------
def save_opts(opt):
     save_dir = opt.log_folder
     to_save = opt.__dict__.copy()

     with open(os.path.join(save_dir, 'opt.json'), 'w') as f:
          json.dump(to_save, f, indent=2)
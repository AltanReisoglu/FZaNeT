import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob
import os 
import os
import torch
import glob

import numpy as np
# from noise import Simplex_CLASS


import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")

classor=os.listdir("weights")[-1]
pth_files = glob.glob(os.path.join(f"weights/{classor}", '*.pth'))



def updater(bn,dfs,student,target,device="cuda"):
    """ bn.load_state_dict(torch.load(pth_files[0], map_location=torch.device(device),weights_only=True))
    dfs.load_state_dict(torch.load(pth_files[1], map_location=torch.device(device),weights_only=True))
    student.load_state_dict(torch.load(pth_files[2], map_location=torch.device(device),weights_only=True))
    target.load_state_dict(torch.load(pth_files[3], map_location=torch.device(device),weights_only=True))"""
    checkpoint_bn = torch.load(pth_files[0],weights_only=True) # Replace with your checkpoint path
    #***************************
    # Get the state_dict of your current model
    model_state_dict_bn = bn.state_dict()

    # Create a new state_dict with only the matching keys and sizes
    pretrained_dict_bn = {k: v for k, v in checkpoint_bn.items() if k in model_state_dict_bn and v.size() == model_state_dict_bn[k].size()}

    # Update your model's state_dict with the matching parameters
    model_state_dict_bn.update(pretrained_dict_bn)

    # Load the updated state_dict into your model
    bn.load_state_dict(model_state_dict_bn)

     #***************************
    checkpoint_dfs = torch.load(pth_files[1],weights_only=True)
    # Get the state_dict of your current model
    model_state_dict_dfs = dfs.state_dict()

    # Create a new state_dict with only the matching keys and sizes
    pretrained_dict_dfs = {k: v for k, v in checkpoint_dfs.items() if k in model_state_dict_dfs and v.size() == model_state_dict_dfs[k].size()}

    # Update your model's state_dict with the matching parameters
    model_state_dict_dfs.update(pretrained_dict_dfs)

    # Load the updated state_dict into your model
    dfs.load_state_dict(model_state_dict_dfs)

    checkpoint_st = torch.load(pth_files[2],weights_only=True)
    # Get the state_dict of your current model
    model_state_dict_st = student.state_dict()

    # Create a new state_dict with only the matching keys and sizes
    pretrained_dict_st = {k: v for k, v in checkpoint_st.items() if k in model_state_dict_st and v.size() == model_state_dict_st[k].size()}

    # Update your model's state_dict with the matching parameters
    model_state_dict_st.update(pretrained_dict_st)

    # Load the updated state_dict into your model
    student.load_state_dict(model_state_dict_st)

    checkpoint_tr = torch.load(pth_files[3],weights_only=True)
    # Get the state_dict of your current model
    model_state_dict_tr = target.state_dict()

    # Create a new state_dict with only the matching keys and sizes
    pretrained_dict_tr = {k: v for k, v in checkpoint_tr.items() if k in model_state_dict_tr and v.size() == model_state_dict_tr[k].size()}

    # Update your model's state_dict with the matching parameters
    model_state_dict_tr.update(pretrained_dict_tr)

    # Load the updated state_dict into your model
    target.load_state_dict(model_state_dict_tr)


    print("weights loaded")
    
    return bn,dfs,student,target

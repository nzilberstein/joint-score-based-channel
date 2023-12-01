#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import sys
import os
import argparse
import copy
sys.path.append('./')

from tqdm import tqdm
import scipy.io as sio
import random
from dotmap import DotMap
from torch.utils.data import DataLoader

from ncsnv2.models        import get_sigmas
from ncsnv2.models.ema    import EMAHelper
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
from ncsnv2.losses        import get_optimizer
from ncsnv2.losses.dsm    import anneal_dsm_score_estimation

from data.loaders import Channels
from data.sample_generator import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train', type=str, default='3GPP')
args = parser.parse_args()

# Always
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True
# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

# Model config
config          = DotMap()
config.device   = 'cuda:0'
# Inner model
config.model.ema           = True
config.model.ema_rate      = 0.999
config.model.normalization = 'InstanceNorm++'
config.model.nonlinearity  = 'elu'
config.model.sigma_dist    = 'geometric'
config.model.num_classes   = 2311 # Number of train sigmas and 'N'
config.model.ngf           = 32

# Optimizer
config.optim.weight_decay  = 0.000 # No weight decay
config.optim.optimizer     = 'Adam'
config.optim.lr            = 0.0001
config.optim.beta1         = 0.9
config.optim.amsgrad       = False
config.optim.eps           = 0.001

# Training
config.training.batch_size     = 32
config.training.num_workers    = 4
config.training.n_epochs       = 40
config.training.anneal_power   = 2
config.training.log_all_sigmas = False
config.training.eval_freq      = 50 # In epochs

# Data
config.data.channel        = args.train # Training and validation
config.data.channels       = 2 # {Re, Im}
config.data.num_pilots     = 8
config.data.noise_std      = 0.01 # 'Beta' in paper
config.data.image_size     = [64, 32] # Channel size = Nr x Nt
config.data.mixed_channels = False
config.data.norm_channels  = 'global' # Optional, no major impact

# Universal seeds
train_seed, val_seed = 1234, 4321

# Get datasets and loaders for channels
train_samples = 7500
mat_contents = sio.loadmat('data/H_bank_64.mat')
H = mat_contents['H_bank']
H_complex = torch.tensor(H[:train_samples, :, :]).detach().numpy()#Pick up NT random users from 100.


dataset     = Channels(train_seed, config, H = H_complex, norm=config.data.norm_channels)
dataloader  = DataLoader(dataset, batch_size=config.training.batch_size, 
                         shuffle=True, num_workers=config.training.num_workers,
                         drop_last=True)


# Validation data
H_val_complex = torch.tensor(H[train_samples:9500, :, :]).detach().numpy()#Pick up NT random users from 100.
val_samples = H_val_complex.shape[0]

val_config = copy.deepcopy(config)
val_datasets = Channels(val_seed, val_config, H = H_val_complex,  norm=config.data.norm_channels)
val_loaders = DataLoader(val_datasets, 
                         batch_size=len(val_datasets),
                         shuffle=False, 
                         num_workers=0, 
                         drop_last=True)
val_iters = iter(val_loaders)


# Pre-determined values
config.model.sigma_begin = 30 # !!! For CDL-C
config.model.sigma_rate = 0.995 # !!! For everything
config.model.sigma_end  = config.model.sigma_begin * config.model.sigma_rate ** (config.model.num_classes - 1)

# Choose the step size (epsilon) according to [Song '20]
candidate_steps = np.logspace(-13, -8, 1000)
step_criterion  = np.zeros((len(candidate_steps)))
gamma_rate      = 1 / config.model.sigma_rate
for idx, step in enumerate(candidate_steps):
    step_criterion[idx] = (1 - step / config.model.sigma_end ** 2) \
        ** (2 * config.model.num_classes) * (gamma_rate ** 2 -
            2 * step / (config.model.sigma_end ** 2 - config.model.sigma_end ** 2 * (
                1 - step / config.model.sigma_end ** 2) ** 2)) + \
            2 * step / (config.model.sigma_end ** 2 - config.model.sigma_end ** 2 * (
                1 - step / config.model.sigma_end ** 2) ** 2)
best_idx = np.argmin(np.abs(step_criterion - 1.))
config.model.step_size = candidate_steps[best_idx]


# Get the model
diffuser = NCSNv2Deepest(config)
diffuser = diffuser.cuda()

# Get optimizer
optimizer = get_optimizer(config, diffuser.parameters())


# Counter
start_epoch = 0
step = 0

if config.model.ema:
    ema_helper = EMAHelper(mu=config.model.ema_rate)
    ema_helper.register(diffuser)

# Get a collection of sigma values
sigmas = get_sigmas(config)

# Always the same initial points and data for validation
val_H_list = []

val_sample = next(val_iters)
# val_H_list.append(val_sample['H_herm'].cuda())

# More logging
config.log_path = 'models/sigmaT%.1f' % (config.model.sigma_begin)

if not os.path.exists(config.log_path):
    os.makedirs(config.log_path)
    
# No sigma logging
hook = test_hook = None

# Logged metrics
train_loss, val_loss  = [], []
val_errors, val_epoch = [], []

for epoch in tqdm(range(start_epoch, config.training.n_epochs)):
    for i, sample in tqdm(enumerate(dataloader)):
        # Safety check
        diffuser.train()
        step += 1
        
        # Move data to device
        for key in sample:
            sample[key] = sample[key].cuda()
        
        # Get loss on Hermitian channels
        loss = anneal_dsm_score_estimation(
            diffuser, sample['H_herm'], sigmas, None, 
            config.training.anneal_power, hook)
        
        # Keep a running loss
        if step == 1:
            running_loss = loss.item()
        else:
            running_loss = 0.99 * running_loss + 0.01 * loss.item()
        # Log
        train_loss.append(loss.item())
        
        # Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # EMA update
        if config.model.ema:
            ema_helper.update(diffuser)
            
        # Verbose
        if step % 100 == 0:
            if config.model.ema:
                val_score = ema_helper.ema_copy(diffuser)
            else:
                val_score = diffuser
            
            # For each validation setup
            local_val_losses = []
            for idx in range(len(config.data.spacing_list)):
                with torch.no_grad():
                    val_dsm_loss = \
                        anneal_dsm_score_estimation(
                            val_score, val_H_list[idx],
                            sigmas, None,
                            config.training.anneal_power,
                            hook=test_hook)
                # Store
                local_val_losses.append(val_dsm_loss.item())
            # Sanity delete
            del val_score
            # Log
            val_loss.append(local_val_losses)
                
            # Print
            if len(local_val_losses) == 1:
                print('Epoch %d, Step %d, Train Loss (EMA) %.3f, Val. Loss %.3f' % (epoch, step, running_loss, 
                                                                                    local_val_losses[0]))
            elif len(local_val_losses) == 2:
                print('Epoch %d, Step %d, Train Loss (EMA) %.3f, Val. Loss (Split) %.3f %.3f' % (epoch, step, running_loss, 
                                                                                                local_val_losses[0], local_val_losses[1]))
        
# Save snapshot
torch.save({'model_state': diffuser.state_dict(),
            'optim_state': optimizer.state_dict(),
            'config': config,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_errors': val_errors,
            'val_epoch': val_epoch}, 
   os.path.join(config.log_path, 'final_model_3gpp_64.pt'))
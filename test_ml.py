#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import sys
import argparse

from tqdm import tqdm as tqdm
from data.loaders import Channels
from torch.utils.data import DataLoader
import scipy.io as sio

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='3gpp')
parser.add_argument('--channel', type=str, default='3gpp')
parser.add_argument('--antennas', nargs='+', type=int, default=[64, 32])
parser.add_argument('--array', type=str, default='ULA')
parser.add_argument('--spacing', nargs='+', type=float, default=[0.5])
parser.add_argument('--alpha', nargs='+', type=float, default=[0.6])
args = parser.parse_args()

# Always !!!
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "";
# All the threading in the world
num_threads = 2
torch.set_num_threads(num_threads)
os.environ["OMP_NUM_THREADS"]        = str(num_threads)
os.environ["OMP_DYNAMIC"]            = "false"
os.environ["OPENBLAS_NUM_THREADS"]   = str(num_threads)
os.environ["MKL_NUM_THREADS"]        = str(num_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"]    = str(num_threads)

# Target weights
target_weights = './models/final_model_3gpp_64x32.pt'
contents = torch.load(target_weights, map_location=torch.device('cpu'))

# Extract config and set some paramaters
config = contents['config']
config.data.channel   = args.model
config.data.channel = args.channel
config.data.num_pilots  = int(30)
config.data.mod_n = 4
NR = config.data.image_size[0]
NT = config.data.image_size[1]
M = int(np.sqrt(config.data.mod_n))
num_channels       = 50 # Validation 

# Sseeds
train_seed, val_seed = 1234, 4321

# Range of SNR, test channels and hyper-parameters
snr_range          = np.asarray(np.arange(-10, 22.5, 2.5))
noise_range        = 10 ** (-snr_range / 10.)

# Global results
oracle_log = np.zeros((len(snr_range), num_channels)) # Should match data
result_dir = './results/results_ml_baseline' % (args.model, args.channel)
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

# Load data
mat_contents = sio.loadmat('data/H_bank_64.mat')
H = mat_contents['H_bank']
H_test_complex = torch.tensor(H[9500:9500 + num_channels, :, :]).detach().numpy()

# Create locals
dataset = Channels(val_seed, config,  H = H_test_complex, norm=[0, 1])
loader  = DataLoader(dataset, batch_size= num_channels, 
                     shuffle=False, num_workers=0, drop_last=True)
iter_ = iter(loader) # For validation
print('There are %d validation channels!' % len(dataset))
    
# Create each variable of the forward model
val_sample = next(iter_)
val_P = val_sample['P']
# Transpose pilots
val_P = torch.conj(torch.transpose(val_P, -1, -2))
val_H_herm = val_sample['H_herm']
val_H = val_H_herm[:, 0] + 1j * val_H_herm[:, 1]

# De-tensorize
val_P = val_P.resolve_conj().numpy()
val_H = val_H.resolve_conj().numpy()

# For each SNR value
for snr_idx, local_noise in tqdm(enumerate(noise_range)):
    y_pilots     = np.matmul(val_P, val_H)
    y_pilots     = y_pilots + np.sqrt(local_noise) * (np.random.normal(size=y_pilots.shape) + 
                                                      1j * np.random.normal(size=y_pilots.shape))
    
    # !!! For each sample
    for sample_idx in tqdm(range(y_pilots.shape[0])):
        # Normal equation
        normal_P = np.matmul(val_P[sample_idx].T.conj(), val_P[sample_idx]) + \
            4 * local_noise * np.eye(val_P[sample_idx].shape[-1])
        normal_Y = np.matmul(val_P[sample_idx].T.conj(), y_pilots[sample_idx])
        # Single-shot solve
        est_H, _, _, _ = np.linalg.lstsq(normal_P, normal_Y)
        
        # Estimate error
        oracle_log[snr_idx, sample_idx] = \
            (np.sum(np.square(
                np.abs(est_H - val_H[sample_idx])), axis=(-1, -2))) / \
                np.sum(np.square(np.abs(val_H[sample_idx])), axis=(-1, -2))
            
# Save to file
torch.save({'snr_range': snr_range,
            'oracle_log': oracle_log
            }, result_dir + \
            '/results_Nt%d_Nr%d_30pilots_lr.pt' % (
                args.antennas[1], args.antennas[0]))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:32:31 2021

@author: marius
"""

import os
# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "";
# All the threading in the world
num_threads = 2
os.environ["OMP_NUM_THREADS"]        = str(num_threads)
os.environ["OMP_DYNAMIC"]            = "false"
os.environ["OPENBLAS_NUM_THREADS"]   = str(num_threads)
os.environ["MKL_NUM_THREADS"]        = str(num_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"]    = str(num_threads)

import numpy as np
import torch, sys, itertools, copy, argparse
sys.path.append('./')

from tqdm import tqdm as tqdm
from data.loaders          import Channels
from torch.utils.data import DataLoader
import scipy.io as sio
import random
torch.set_num_threads(num_threads)

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

# Target weights
# target_weights = './models/\
# numLambdas1_lambdaMin0.5_lambdaMax0.5_sigmaT30.0/final_model_kronecker_64nr.pt'
target_weights = './models/\
numLambdas1_lambdaMin0.5_lambdaMax0.5_sigmaT30.0/final_model_3gpp_64.pt'
contents = torch.load(target_weights, map_location=torch.device('cpu'))
# Extract config
config = contents['config']
config.sampling.sigma = 0. # Nothing here
config.data.channel   = args.model # Always for training!
# More stuff
config.data.array        = args.array
config.data.image_size   = [args.antennas[0], args.antennas[1]]
config.data.spacing_list = [args.spacing[0]]
# Universal seeds
train_seed, val_seed = 1234, 4321
# Get training config
# dataset = Channels(train_seed, config, norm=config.data.norm_channels)

# Range of SNR, test channels and hyper-parameters
snr_range          = np.asarray(np.arange(-10, 22.5, 2.5))
# snr_range          = np.asarray(np.arange(-30, 17.5, 2.5))
spacing_range      = np.asarray(args.spacing) # From a pre-defined index
alpha_range        = np.asarray(args.alpha)
noise_range        = 10 ** (-snr_range / 10.)

# Limited number of samples
kept_samples = 50

# Global results
oracle_log = np.zeros((len(spacing_range), len(alpha_range),
                       len(snr_range), kept_samples)) # Should match data
result_dir = './results/results_ml_baseline' % (
    args.model, args.channel)
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
    
# Wrap sparsity, steps and spacings
meta_params = itertools.product(spacing_range, alpha_range)

# For each hyper-combo
for meta_idx, (spacing, alpha) in tqdm(enumerate(meta_params)):
    # Unwrap indices
    spacing_idx, alpha_idx, = np.unravel_index(
        meta_idx, (len(spacing_range), len(alpha_range)))
    
    # Get a validation dataset and adjust parameters
    val_config = copy.deepcopy(config)
    # !!! Replace channel
    pilot_alpha = [32/64]
    val_config.purpose      = 'val'
    val_config.data.channel = args.channel
    val_config.data.spacing_list = [spacing]
    val_config.data.num_pilots   = \
        int(30)
    val_config.data.mod_n = 4
    

    NR = val_config.data.image_size[0]
    NT = val_config.data.image_size[1]
    M = int(np.sqrt(val_config.data.mod_n))
    num_channels       = 50 # Validation 
    val_samples        = num_channels

    our_dir = 'results_seed4321'

    # our_file = our_dir + '/final_experiments/%2sx%2s/batch/%s_numpilots%.1f_numsymbols%.1f_%1sjointstepxnoiselevel_reuse_batch2.pt' % \
    #             (NR, NT, args.channel, 32, 32, 3)
    # dictionary_H_kron = torch.load(our_file)
    # H_val_complex = dictionary_H_kron['H_val_complex']

    mat_contents = sio.loadmat('data/H_bank_64.mat')
    H = mat_contents['H_bank']
    # H = torch.tensor(H[:, :, 0:config.NT])
    H_val_complex = torch.tensor(H[9500:9500 + val_samples, :, :]).detach().numpy()#Pick up NT random users from 100.
    # H_val_complex = torch.tensor(H[:, :, :]).detach().numpy()#Pick up NT random users from 100.
    

    # Create locals
    val_dataset = Channels(val_seed, val_config,  H = H_val_complex, norm=[0, 1])
    # val_dataset = Channels(val_seed, val_config, norm=config.data.norm_channels)
    batch_size = len(val_dataset)
    val_loader  = DataLoader(val_dataset, batch_size= val_samples,
        shuffle=False, num_workers=0, drop_last=True)
    val_iter = iter(val_loader) # For validation
    print('There are %d validation channels!' % len(val_dataset))
        
    # Always the same initial points and data for validation
    val_sample = next(val_iter)
    del val_iter, val_loader
    val_P = val_sample['P']
    # Transpose pilots
    val_P = torch.conj(torch.transpose(val_P, -1, -2))
    val_H_herm = val_sample['H_herm']
    val_H = val_H_herm[:, 0] + 1j * val_H_herm[:, 1]
    
    # De-tensorize
    val_P = val_P.resolve_conj().numpy()
    val_H = val_H.resolve_conj().numpy()
    
    # Keep only relevant samples
    val_P = val_P[:kept_samples, ...]
    val_H = val_H[:kept_samples, ...]
    
    # For each SNR value
    for snr_idx, local_noise in tqdm(enumerate(noise_range)):
        val_Y     = np.matmul(val_P, val_H)
        val_Y     = val_Y + \
            np.sqrt(local_noise) * \
                (np.random.normal(size=val_Y.shape) + \
                 1j * np.random.normal(size=val_Y.shape))
        
        # !!! For each sample
        for sample_idx in tqdm(range(val_Y.shape[0])):
            # Normal equation
            normal_P = np.matmul(val_P[sample_idx].T.conj(), val_P[sample_idx]) + \
                4 * local_noise * np.eye(val_P[sample_idx].shape[-1])
            normal_Y = np.matmul(val_P[sample_idx].T.conj(), val_Y[sample_idx])
            # Single-shot solve
            est_H, _, _, _ = np.linalg.lstsq(normal_P, normal_Y)
            
            # Estimate error
            oracle_log[spacing_idx, alpha_idx, snr_idx, sample_idx] = \
                (np.sum(np.square(
                    np.abs(est_H - val_H[sample_idx])), axis=(-1, -2)))/\
                 np.sum(np.square(
                     np.abs(val_H[sample_idx])), axis=(-1, -2))
                
# Save to file
torch.save({'snr_range': snr_range,
            'spacing_range': spacing_range,
            'alpha_range': alpha_range,
            'oracle_log': oracle_log
            }, result_dir + \
            '/results_Nt%d_Nr%d_30pilots_lr.pt' % (
                args.antennas[1], args.antennas[0]))
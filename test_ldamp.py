#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, copy, itertools, argparse
sys.path.append('..')
import numpy as np

import torch
from aux_models import LDAMP
from tqdm import tqdm
import scipy.io as sio


from data.loaders          import Channels
from torch.utils.data import DataLoader
from matplotlib       import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train', type=str, default='CDL-C')
parser.add_argument('--test', type=str, default='3gpp')
parser.add_argument('--snr_range', nargs='+', type=float, 
                    default=np.arange(-10, 22.5, 2.5))
args = parser.parse_args()

# Disable TF32 due to potential precision issues
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True
# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Model backbone
train_backbone = 'FlippedUNet'

# Channel antenna spacing and pilot alpha
spacing_range     = [0.5]
pilot_alpha_range = [0.6]
snr_range         = args.snr_range

# Training and testing seeds
train_seed, test_seed = 1234, 4321

# Wrap spacing, sparsity and SNR
meta_params = itertools.product(spacing_range, pilot_alpha_range, snr_range)

# Number of validation channels and result logging
num_channels = 50
nmse_log     = np.zeros((len(spacing_range), len(pilot_alpha_range),
                         len(snr_range), num_channels))
# Result directory
result_dir = './results/results_ldamp/train-%s_test-%s' % (
    args.train, args.test)
os.makedirs(result_dir, exist_ok=True)

# For each hyper-combo
for meta_idx, (spacing, pilot_alpha, snr) in tqdm(enumerate(meta_params)):
    # Unwrap indices
    spacing_idx, pilot_alpha_idx, snr_idx = np.unravel_index(
        meta_idx, (len(spacing_range), len(pilot_alpha_range),
                   len(snr_range)))
    
    # Load model weights
    target_dir = './models/ldamp-%s/train-%s' % (
        train_backbone, args.train)
    target_file = os.path.join(target_dir, 'model_snr%.2f_alpha%.2f.pt' % (
        snr, pilot_alpha))
    contents = torch.load(target_file)
    config   = contents['config']
    
    # Create a model (just once) and load weights (every SNR point)
    if meta_idx == 0:
        model = LDAMP(config.model)
        model = model.cuda()
    model.load_state_dict(contents['model_state'])
    model.eval()
    
    # Get a validation dataset and adjust parameters
    val_config = copy.deepcopy(config)
    val_config.data.channel      = args.test
    val_config.data.spacing_list = [spacing]
    val_config.data.train_pilots = config.data.train_pilots
    val_config.data.train_snr    = np.asarray([snr])
    val_config.data.noise_std    = 10 ** (-val_config.data.train_snr / 10.)
    
    # Get training dataset for normalization purposes
    if meta_idx == 0:
        train_dataset = Channels(
            train_seed, config, norm=config.data.norm_channels)
        norm        = [train_dataset.mean, train_dataset.std]
        
        
    val_samples        = num_channels

    our_dir = 'results_seed4321'

    mat_contents = sio.loadmat('data/H_bank_64.mat')
    H = mat_contents['H_bank']
    # H = torch.tensor(H[:, :, 0:config.NT])
    H_val_complex = torch.tensor(H[9500:9500 + val_samples, :, :]).detach().numpy()#Pick up NT random users from 100.
    # H_val_complex = torch.tensor(H[:, :, :]).detach().numpy()#Pick up NT random users from 100.

    

    # Create locals
    dataset = Channels(test_seed, val_config,  H = H_val_complex, norm=[0, 1])
    dataloader  = DataLoader(dataset, batch_size=num_channels, shuffle=False)
    
    # For each batch of samples
    for batch_idx, sample in tqdm(enumerate(dataloader)):
        # Move samples to GPU
        for key in sample.keys():
            sample[key] = sample[key].cuda()
        
        # Estimate channels
        with torch.no_grad():
            H_est = model(sample, config.model.max_unrolls)

            # Compute NMSE
            nmse_loss = \
                torch.sum(torch.square(torch.abs(H_est - sample['H_herm_cplx'])),
                          dim=(-1, -2))/\
                torch.sum(torch.square(torch.abs(sample['H_herm_cplx'])), 
                          dim=(-1, -2))
        
        # Store NMSE for each channel instance
        nmse_log[spacing_idx, pilot_alpha_idx, snr_idx] = \
            nmse_loss.cpu().detach().numpy()
                
# Get average NMSE
avg_nmse = np.mean(nmse_log, axis=-1)
# For each alpha and SNR value
for alpha_idx, local_alpha in enumerate(pilot_alpha_range):
    for snr_idx, local_snr in enumerate(snr_range):
        local_nmse = avg_nmse[0, alpha_idx, snr_idx]
        # Print result
        print('Learned D-AMP: SNR = %.2f dB, NMSE = %.2f dB' % (
            local_snr, 10*np.log10(local_nmse)))

# Plot results for all alpha values
plt.rcParams['font.size'] = 14
plt.figure(figsize=(10, 10))
for alpha_idx, local_alpha in enumerate(pilot_alpha_range):
    plt.plot(snr_range, 10*np.log10(avg_nmse[0, alpha_idx]),
             linewidth=4, label='Alpha=%.2f' % local_alpha)
plt.grid(); plt.legend()
plt.title('Learned Denoising AMP')
plt.xlabel('SNR [dB]'); plt.ylabel('NMSE [dB]')
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'results.png'), dpi=300, 
            bbox_inches='tight')
plt.close()

# Save full results to file
torch.save({'nmse_log': nmse_log,
            'avg_nmse': avg_nmse,
            'snr_range': snr_range,
            'pilot_alpha_range': pilot_alpha_range,
            'spacing_range': spacing_range,
            'config': config, 'args': args
            }, os.path.join(result_dir, 'results.pt'))
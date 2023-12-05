import numpy as np
import torch
import sys
import os
import copy
import argparse
import scipy.io as sio
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm
sys.path.append('./')

from ncsnv2.models.ncsnv2 import NCSNv2Deepest
from data.loaders          import Channels
from torch.utils.data import DataLoader
from utils.util import *
from data.sample_generator import *

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--channel', type=str, default='3GPP')
parser.add_argument('--save_channels', type=int, default=0)
parser.add_argument('--pilot_alpha', nargs='+', type=float, default=[32/64])
parser.add_argument('--noise_boost', nargs='+', type=float, default=0.001)
parser.add_argument('--batch_size_x_list', nargs='+', type=float, default=[50])
parser.add_argument('--pilots_list', nargs='+', type=float, default=[30])
parser.add_argument('--sample_joint', type=bool, default=True)

args = parser.parse_args()


# logger
logger = get_logger()

# Cuda config
torch.cuda.empty_cache()
device = 'cuda:0'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark = True
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu);    

logger.info(f"Device set to {device}.")

# Load configs
test_seed = 4321
result_dir = 'results_joint_seed%d' % test_seed
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
logger.info(f"Results will be saved to {result_dir}.")

target_weights = './models/model_3gpp_64x32.pt'
contents = torch.load(target_weights)

config = contents['config']
config.sampling.steps_each = 3
config.data.channel  = args.channel
config.model.step_size = 1 * 1e-10
config.data.mod_n = 4

# Get and load the model
diffuser = NCSNv2Deepest(config)
diffuser = diffuser.cuda()
diffuser.load_state_dict(contents['model_state']) 
diffuser.eval()

# Set some paramaters
snr_range = np.arange(-10, 17.5, 2.5)
noise_range = 10 ** (-snr_range / 10.)

NR = config.data.image_size[0]
NT = config.data.image_size[1]
M = int(np.sqrt(config.data.mod_n))

num_channels = 50 
total_iter = int(config.model.num_classes * config.sampling.steps_each) 
noise_boost = args.noise_boost
logger.info(f"Size of the channels: {NR}x{NT}.")
logger.info(f"Total number of iterations: {total_iter}.")


# Load 3gpp channels and prepare dataloader for symbol estimation
mat_contents = sio.loadmat('data/H_bank_64.mat')
H = mat_contents['H_bank']
H_test_complex = torch.tensor(H[9500:9500 + num_channels, :, :]).detach().numpy() #Pick up NT random users from 100.

# Create generator for the symbols
generator = sample_generator(num_channels, config.data.mod_n, NR)

# Convert matrix to real representation
aux = torch.tensor(H_test_complex)
H_test_real_repr = torch.empty([num_channels, 2 * NR, 2 * NT])
H_test_real_repr[:,0:NR,0:NT] = torch.real(aux)
H_test_real_repr[:,0:NR,NT:] = torch.imag(aux)
H_test_real_repr[:,NR:,0:NT] = torch.imag(aux)
H_test_real_repr[:,NR:,NT:] = torch.real(aux)
H_test_real_repr[:,:NR,NT:] = -H_test_real_repr[:,:NR,NT:]

logger.info(f"Channels loaded.")


# Main loop -- inference

for batch_size_x in args.batch_size_x_list:  
    for pilots in args.pilots_list:
        logger.info(f"Starting experiment with this number of pilots: {pilots}.")
        logger.info(f"Starting experiment with this number of symbols: {batch_size_x}.")
        
        # Set some hyperparameters
        SER_langevin = []
        oracle_log = np.zeros((len(snr_range), total_iter)) 
        config.data.num_pilots = pilots
        print(config.data.num_pilots)

        # Load data
        dataset_pilots = Channels(test_seed, config, H = H_test_complex, norm="global")
        batch_size = len(dataset_pilots)
        loader  = DataLoader(dataset_pilots, batch_size= num_channels, 
                             shuffle=False, num_workers=0, drop_last=True)
        
        iter_ = iter(loader) 
        samples_pilots = next(iter_)
        _, pilots, _ = samples_pilots['H'].cuda(), samples_pilots['P'].cuda(), samples_pilots['Y'].cuda()

        pilots_conj = torch.conj(torch.transpose(pilots, -1, -2))
        H_herm = samples_pilots['H_herm'].cuda()
        H_herm_complex = H_herm[:, 0] + 1j * H_herm[:, 1]
        
        # Start the loop for all SNRs
        for snr_idx, local_noise in enumerate(noise_range):
            
            # Setting parameters for each SNR
            iter_lang = 0
            Id = batch_identity_matrix(2 * NR, 2 * NR, batch_size)
            if snr_range[snr_idx] < 5:
                temp_x      = 0.5 #0.7
                sigmas_x    = np.linspace(0.6, 0.01, config.model.num_classes)
                epsilon     = 1E-4
            else:
                temp_x      = 0.1
                sigmas_x    = np.linspace(0.8, 0.01, config.model.num_classes)
                epsilon     = 4E-5
                        
            # Prepare data associated to the pilots
            y_pilots       = torch.matmul(pilots_conj, H_herm_complex)
            y_pilots     = y_pilots + np.sqrt(local_noise) * torch.randn_like(y_pilots) 
            H_current = torch.randn_like(H_herm_complex)
            oracle    = H_herm_complex
            H_list = []

            # Prepare data associated to the symbols
            x_current = ((1 + 1) * torch.rand(num_channels, 2 * NT, batch_size_x) + 1).to(device=device)
            indices   = generator.random_indices(NT, batch_size_x * num_channels)
            j_indices = generator.joint_indices(indices)
            x_true    = generator.modulate(indices)
            x_true    = torch.reshape(x_true, (num_channels, batch_size_x, 2 * NT)) 
            x_true    = torch.transpose(x_true, -1, -2)
            y_x       = torch.matmul(H_test_real_repr.double(), x_true.double()).to(device=device).float()
            y_x       = y_x + np.sqrt(local_noise) * torch.randn_like(y_x).to(device=device)
            H_current_x = torch.zeros([num_channels, 2 * NR, 2 * NT])

            # Create joint vector of measurements
            y_x_complex = y_x.chunk(2, dim =1)
            y_x_complex = (y_x_complex[0] - 1j * y_x_complex[1])
            y_x_complex = torch.transpose(y_x_complex, -1 , -2)
            y_H = torch.cat((y_pilots.to(device=device), y_x_complex.to(device=device)), dim = 1)

            with torch.no_grad():
                for step_idx in tqdm(range(config.model.num_classes)):
                    # Compute current step size and noise power
                    current_sigma = diffuser.sigmas[step_idx].item()
                    current_sigma_x = sigmas_x[step_idx]
                    
                    # Labels for diffusion model
                    labels = torch.ones(H_current.shape[0]).cuda() * step_idx
                    labels = labels.long()

                    # Step size for each dynamic
                    step_H = config.model.step_size * \
                            (current_sigma / config.model.sigma_end) ** 2
                    step_x = epsilon * \
                            (current_sigma_x / sigmas_x[-1]) ** 2    #7E-5

                    # For each step spent at that noise level
                    for inner_idx in range(config.sampling.steps_each):
                    
                        H_current_nonHerm = torch.transpose(torch.conj(H_current), 2, 1).to(device=device)
                        H_current_x[:,0:NR,0:NT] = torch.real(H_current_nonHerm)
                        H_current_x[:,0:NR,NT:] = torch.imag(H_current_nonHerm)
                        H_current_x[:,NR:,0:NT] = torch.imag(H_current_nonHerm)
                        H_current_x[:,NR:,NT:] = torch.real(H_current_nonHerm)
                        H_current_x[:,NR:,0:NT] = -H_current_x[:,NR:,0:NT]
                        
                        #------------------------#
                        # Compute Langevin for x #
                        #------------------------#

                        grad = torch.zeros((num_channels, 2 * NT, batch_size_x)).to(device=device)
                        # Score of the prior
                        x_gaussian = torch.transpose(x_current, 2, 1)
                        Zi_hat = gaussian(x_gaussian.reshape(batch_size_x * num_channels, 2 *NT), generator, current_sigma_x**2, NT, M, device)
                        Zi_hat = torch.transpose(torch.reshape(Zi_hat, (num_channels, batch_size_x, 2 * NT)), 2, 1)
                        prior = (Zi_hat - x_current) / current_sigma_x**2
                        
                        # Score of the likelihood
                        diff =  (y_x - torch.matmul(H_current_x.to(device=device), x_current))
                        cov_matrix = (current_sigma_x**2) * torch.bmm(H_current_x, torch.transpose(H_current_x, 2, 1)) + local_noise * Id
                        cov_matrix = torch.inverse(cov_matrix.to(device=device))
                        grad_likelihood = torch.matmul(cov_matrix, diff.float()).to(device=device)
                        grad_likelihood = torch.matmul(torch.transpose(H_current_x, 2, 1).to(device=device), grad_likelihood)
                        del cov_matrix

                        # Score of the posterior
                        grad = grad_likelihood + prior

                        # Noise generation
                        noise = np.sqrt( 2 * temp_x * step_x) * torch.randn(num_channels, 2 * NT, batch_size_x).to(device=device)
                        
                        # Update
                        x_current = x_current + step_x * grad + noise
        
                        #------------------------#
                        # Compute Langevin for H #
                        #------------------------#

                        #  Score of the prior
                        current_real = torch.view_as_real(H_current).permute(0, 3, 1, 2)
                        # Get score
                        score = diffuser(current_real, labels)
                        # View as complex
                        score = torch.view_as_complex(score.permute(0, 2, 3, 1).contiguous())

                        # Score of the likelihood
                        if args.sample_joint == True:
                            forward_complex = x_current.chunk(2, dim = 1)
                            forward_complex = (forward_complex[0] - 1j * forward_complex[1])
                            forward_complex = torch.transpose(forward_complex, -1 , -2)
                            forward_H = torch.cat((pilots_conj.to(device=device), forward_complex.to(device=device)), dim = 1)
                            forward_herm = torch.conj(torch.transpose(forward_H, -1, -2)).to(device=device)
                            meas_grad = torch.matmul(forward_herm, 
                                                    torch.matmul(forward_H,  H_current.to(device=device)) - y_H
                                                    )
                        else:
                            meas_grad = torch.matmul(pilots, 
                                                    torch.matmul(pilots_conj, H_current.to(device=device)) - y_pilots
                                                    )

                        # Noise generation
                        grad_noise = np.sqrt(2 * step_H * noise_boost) * torch.randn_like(H_current) 
                        
                        # Update
                        H_current = H_current.to(device=device) \
                                    + step_H * (score.to(device=device) - meas_grad.to(device=device) / (local_noise/2. + current_sigma ** 2)) \
                                    + grad_noise.to(device=device)
                        
                        # Store error
                        oracle_log[snr_idx, iter_lang] = \
                            torch.mean((torch.sum(torch.square(torch.abs(H_current.to(device='cpu') - oracle.to(device='cpu'))), dim=(-1, -2))/\
                            torch.sum(torch.square(torch.abs(oracle.to(device='cpu'))), dim=(-1, -2)))).cpu().numpy()
                        iter_lang = iter_lang + 1

            
            H_list.append(H_current_x)
            SER_langevin.append(1 - sym_detection(torch.transpose(x_current, -1, -2).reshape(num_channels * batch_size_x, 2 * NT).to(device='cpu'), j_indices, generator.real_QAM_const, generator.imag_QAM_const))
            print(snr_range[snr_idx], 10 * np.log10(oracle_log[:,-1]))

        torch.cuda.empty_cache()

        # Save results to file based on noise
        save_dict = {
                    'snr_range': snr_range,
                    'config': config,
                    'oracle_log': oracle_log,
                    'H_val_complex': H_test_complex,
                    'H_symbols_batch': H_test_real_repr,
                    'H_current_x': H_list,
                    'SER_langevin': SER_langevin
                    }   
        torch.save(save_dict,
                   result_dir + '/%s_numpilots%.1f_numsymbols%.1f.pt' % (args.channel, config.data.num_pilots, batch_size_x))
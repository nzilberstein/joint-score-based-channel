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
import yaml
sys.path.append('./')

from ncsnv2.models.ncsnv2 import NCSNv2Deepest
from data.loaders          import Channels
from torch.utils.data import DataLoader
from utils.logger import get_logger
from utils.util import *
from data.sample_generator import *

from data.sample_generator import *

parser = argparse.ArgumentParser()
parser.add_argument('--use_GPU', type=bool, default=True)
parser.add_argument('--channel', type=str, default='3gpp')
parser.add_argument('--pilots_list', nargs='+', type=int, default=[30])
parser.add_argument('--batch_symbols_x_list', nargs='+', type=int, default=[50])
parser.add_argument('--sample_joint', type=bool, default=True)
args = parser.parse_args()

# logger
logger = get_logger()

# Device setting
if args.use_GPU and torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = 'cuda:0'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu);         
else:
    device = 'cpu'

logger.info(f"Device set to {device}.")

# Set paths and seed
train_seed, val_seed = 1234, 4321
result_dir = 'results_seed%d' % val_seed
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
our_dir = 'results_paper_seed4321'
logger.info(f"Results will be saved to {result_dir}.")
  

# Load Model and parameters
diffusion_model = './models/numLambdas1_lambdaMin0.5_lambdaMax0.5_sigmaT30.0/final_model_3gpp_64.pt'
contents = torch.load(diffusion_model)
# Extract config and load model
config = contents['config']
val_config = copy.deepcopy(config)
# Get and load the model
diffuser = NCSNv2Deepest(config)
diffuser = diffuser.cuda()
diffuser.load_state_dict(contents['model_state']) 
diffuser.eval()
# Load parameters for testing
dirPath = str(Path(os.path.dirname(__file__)))
with open('config.yml', 'r') as f:
    aux = yaml.load(f,  Loader=yaml.FullLoader)
config_sampling = dict2namespace(aux)


# Set parameters of the samping and the system
snr_range         = config_sampling.snr_range
noise_range        = 10 ** (-config_sampling.snr_range / 10.)

NR = val_config.data.image_size[0]
NT = val_config.data.image_size[1]
num_channels = 50 

steps_per_noise_level = config_sampling.steps_per_noise_level
total_iter = int(config.model.num_classes * steps_per_noise_level) 

temp_H = config_sampling.temp_H
epsilon_H = config_sampling.epsilon_H

logger.info(f"There are {num_channels} of size of the channels: {NR}x{NT}.")
logger.info(f"Total number of Langevin iterations: {total_iter}.")


# Load 3gpp channels and generator of the symbols
mat_contents = sio.loadmat('data/H_bank_64.mat')
H = mat_contents['H_bank']
H_val_complex = torch.tensor(H[9500:9500 + num_channels, :, :]).detach().numpy()#Pick up NT random users from 100.

generator = sample_generator(num_channels, config_sampling.mod_n, val_config.data.image_size[0])
aux = torch.tensor(H_val_complex)
H_symbols_batch = torch.empty([num_channels, 2 * NR, 2 * NT])
H_symbols_batch[:,0:NR,0:NT] = torch.real(aux)
H_symbols_batch[:,0:NR,NT:] = torch.imag(aux)
H_symbols_batch[:,NR:,0:NT] = torch.imag(aux)
H_symbols_batch[:,NR:,NT:] = torch.real(aux)
H_symbols_batch[:,:NR,NT:] = -H_symbols_batch[:,:NR,NT:]

logger.info(f"Channels loaded.")


# Start the loop for all different combinations of symbols and pilots
for batch_size_x in args.batch_symbols_x_list:  
    for pilots in args.pilots_list:
        logger.info(f"Starting inference with {pilots} pilots and {batch_size_x} symbols.")
        
        # Define the two vectors to store the results
        SER_langevin = []
        oracle_log = np.zeros((len(snr_range), total_iter))


        # Define batch id matrix and number of pilots
        Id = batch_identity_matrix(2 * NR, 2 * NR, num_channels)
        val_config.data.num_pilots = pilots
        

        # Load data
        dataset_pilots = Channels(val_seed, val_config, 
                                  H = H_val_complex, norm="global")
        loader = DataLoader(dataset_pilots, batch_size = num_channels,
                                shuffle=False, num_workers=0, drop_last=True)
        iter_ = iter(loader) 
        samples_pilots = next(iter_)

        pilots = samples_pilots['P'].cuda()
        pilots_conj = torch.conj(torch.transpose(pilots, -1, -2))

        H_herm = samples_pilots['H_herm'].cuda()
        H_herm_complex = H_herm[:, 0] + 1j * H_herm[:, 1]
        
        
        # Start the loop for all SNRs
        for snr_idx, local_noise in enumerate(noise_range):
            logger.info(f"Starting inference for an SNR of {snr_range[snr_idx] + 10 * np.log10(NT)}dB .")
            
            iter_lang = 0


            # Setting parameters for sampling of x
            if snr_range[snr_idx] < 5:
                temp_x      = config_sampling.temp_x_low_SNR #0.7
                sigmas_x    = np.linspace(0.6, 0.01, config.model.num_classes)
                epsilon_x     = config_sampling.epsilon_x_low_SNR 
            else:
                temp_x      = config_sampling.temp_x_high_SNR
                sigmas_x    = np.linspace(0.8, 0.01, config.model.num_classes)
                epsilon_x     = config_sampling.epsilon_x_high_SNR 
                        

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
            y_x       = torch.matmul(H_symbols_batch.double(), x_true.double()).to(device=device).float()
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
                    step_H = epsilon_H * (current_sigma / val_config.model.sigma_end) ** 2
                    step_x = epsilon_x * (current_sigma_x / sigmas_x[-1]) ** 2


                    # For each step spent at that noise level
                    for inner_idx in range(steps_per_noise_level):
                    
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
                        Zi_hat = gaussian(x_gaussian.reshape(batch_size_x * num_channels, 2 *NT), generator, current_sigma_x**2, NT, config_sampling.M, device)
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
                        grad_noise = np.sqrt(2 * step_H * temp_H) * torch.randn_like(H_current) 
                        

                        # Update
                        H_current = H_current.to(device=device) \
                                    + step_H * (score.to(device=device) - meas_grad.to(device=device) / (local_noise/2. + current_sigma ** 2)) \
                                    + grad_noise.to(device=device)
                        

                        # Store NMSE
                        oracle_log[snr_idx, iter_lang] = \
                            torch.mean((torch.sum(torch.square(torch.abs(H_current.to(device='cpu') - oracle.to(device='cpu'))), dim=(-1, -2))/\
                            torch.sum(torch.square(torch.abs(oracle.to(device='cpu'))), dim=(-1, -2)))).cpu().numpy()
                        iter_lang = iter_lang + 1

            # Store SER
            SER_langevin.append(1 - sym_detection(torch.transpose(x_current, -1, -2).reshape(num_channels * batch_size_x, 2 * NT).to(device='cpu'), j_indices, generator.real_QAM_const, generator.imag_QAM_const))
            logger.info(f"SNR: {snr_range[snr_idx]}, SER: {10 * np.log10(oracle_log[:,-1])}.")
        
        torch.cuda.empty_cache()

        # Save results to file based on noise
        save_dict = {
                    'snr_range': snr_range,
                    'val_config': val_config,
                    'H_val_complex': H_val_complex,
                    'H_symbols_batch': H_symbols_batch,
                    'H_current_x': H_current_x,
                    'oracle_log': oracle_log,
                    'SER_langevin': SER_langevin
                    }   
        
        torch.save(save_dict,
                    result_dir + '/final_experiments/%schannel_numpilots%.1f_numsymbols%.1f.pt' % \
                    (args.channel, val_config.data.num_pilots, batch_size_x))
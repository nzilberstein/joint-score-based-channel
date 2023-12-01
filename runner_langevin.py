################################################################################
#
####                               IMPORTING
#
################################################################################

#\\Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import sys
import pickle as pkl
from numpy import linalg as LA

#\\Own libraries
from model.classic_detectors import *
from data.sample_generator import *
from utils.util import *
from model.langevin_SVD import *
from pypapi import events, papi_high as high


################################################################################
#
####                               MAIN RUN
#
################################################################################
def runLangevin(config, generator, batch_size, device, H_true, H = None):
    #########################################################
    ## Variables definition ## 
    #########################################################

    #Create noise vector for Langevin
    sigma_gaussian = np.exp(np.linspace(np.log(config.sigma_0), np.log(config.sigma_L),config.n_sigma_gaussian))

    #Define list to save data
    SER_lang32u = []

    #Create model
    langevin = Langevin(sigma_gaussian, generator, config.n_sample_init, config.step_size, device)

    start = time.time()


    #########################################################
    ## Main loop ## 
    #########################################################

    for snr in range(0, len(config.SNR_dBs[config.NT])):
        print(config.SNR_dBs[config.NT][snr])
        # Create variables to save each trajectory and for each snr
        dist = torch.zeros((batch_size,config.n_traj))
        list_traj = torch.zeros((batch_size, 2*config.NT, config.n_traj))
        x_hat = torch.zeros((batch_size, 2*config.NT))


        ########################
        #  Samples generation  #
        ########################
        #Generate data      
        # H, y, x, j_indices, noise_sigma = generator.give_batch_data(config.NT,  snr_db_min=config.SNR_dBs[config.NT][snr],
        #                                                              snr_db_max=config.SNR_dBs[config.NT][snr], 
        #                                                                batch_size=batch_size,
        #                                                                correlated_flag=True, 
        #                                                                 rho=0.6)
                                                                        
        y, x, j_indices, noise_sigma = generator.give_batch_data_Hinput(H_true, config.NT, snr_db_min=config.SNR_dBs[config.NT][snr],
                                                                     snr_db_max=config.SNR_dBs[config.NT][snr], 
                                                                     batch_size = batch_size)
        y = y.to(device=device)

        ###############################
        ##  Langevin detector  ##
        ###############################
        for jj in range(0, config.n_traj):
            # start = time.time()
            #run langevin
            # print(singulars * noise_sigma[0])
            sample_last, samples = langevin.forward(H, 
                                                    y, 
                                                    noise_sigma[0], 
                                                    batch_size,
                                                    config.NT, config.M, config.NR)
            
            #Generate n_traj realizations of Langevin and then choose the best one w.r.t to ||y-Hx||^2
            list_traj[:,:,jj] = torch.clone(sample_last)
            dist[:, jj] = torch.norm(y.to(device=device) - batch_matvec_mul(H.to(device=device).float(),
                                                             sample_last.to(device=device)), 2, dim = 1)

            print('Trajectory:', jj)

        end = time.time()
        print(end - start)

        #Pick the best trajectory for each user
        idx = torch.argsort(dist, dim=1, descending = False)

        for nn in range(0, batch_size):
            x_hat[nn, :] = torch.clone(list_traj[nn,:,idx[nn,0]])

        SER_lang32u.append(1 - sym_detection(x_hat.to(device='cpu'), j_indices, generator.real_QAM_const, generator.imag_QAM_const))
        print(SER_lang32u)

    return SER_lang32u, samples, x
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
from model.langevin_SVDaprox import *
from pypapi import events, papi_high as high


################################################################################
#
####                               MAIN RUN
#
################################################################################
def runLangevin(config, generator, batch_size, device, H = None):
    #########################################################
    ## Variables definition ## 
    #########################################################

    #Create noise vector for Langevin
    sigma_gaussian = np.exp(np.linspace(np.log(config.sigma_0), np.log(config.sigma_L),config.n_sigma_gaussian))

    #Define list to save data
    SER_lang32u = []

    #Create model
    langevin = Langevin(sigma_gaussian, generator, config.n_sample_init, config.step_size, device)

    # #Perturbe H
    # wr = torch.empty((batch_size, config.NR, config.NT)).normal_(mean=0.0, std=1./np.sqrt(2.))
    # wi = torch.empty((batch_size, config.NR, config.NT)).normal_(mean=0.0, std=1./np.sqrt(2.))
    # w1 = torch.cat((wr, -1. * wi), dim=2)
    # w2 = torch.cat((wi, wr), dim=2)
    # W_matrix = torch.cat((w1, w2), dim=1)

    # # SNR
    # H_powerdB = 10. * torch.log(torch.mean(torch.sum(H.pow(2), dim=1), dim=0)) / np.log(10.)
    # average_H_powerdB = torch.mean(H_powerdB)

    # W_matrix *= 0.05
    # # W_matrix *= torch.pow(10., (average_H_powerdB - 20 - 10.*np.log10(config.NT))/20.)
    # # print(torch.pow(10., (average_H_powerdB - 20 - 10.*np.log10(config.NT))/20.))
    # H_pert = H + W_matrix

    # hb = 10
    # d0 = 100
    # d = 200
    # gamma = (4 - 0.0065 * hb + 17.1 / hb) + 0.75 * np.random.randn()
    # s = np.random.randn() * (9.6 + 3 * np.random.randn())
    # PL = 78 + 10 * gamma * np.log10(d / d0) + s
    # sigma_NLOS = 10**(-s/10)
    # # gamma = gamma
    # # sigma_NLOS = (d / d0)**(-gamma)

    # nLOS_users = torch.tensor(np.random.choice([0, 1], size=(batch_size, config.NT), p=[1./3, 2./3])).float()
    # Hr = torch.empty((batch_size, config.NR, config.NT)).normal_(mean=0,std=np.sqrt(sigma_NLOS/2))
    # Hi = torch.empty((batch_size, config.NR, config.NT)).normal_(mean=0,std=np.sqrt(sigma_NLOS/2))

    # for ii in range(config.NT):
    #     Hr[:,:,ii] = torch.mul(Hr[:,:,ii], nLOS_users[:,ii].unsqueeze(-1).repeat(1, config.NR))
    #     Hi[:,:,ii] = torch.mul(Hi[:,:,ii], nLOS_users[:,ii].unsqueeze(-1).repeat(1, config.NR))

    # h1 = torch.cat((Hr, -1. * Hi), dim=2)
    # h2 = torch.cat((Hi, Hr), dim=2)
    # H_NLOS = torch.cat((h1, h2), dim=1)

    # H = H + H_NLOS

    #############
    ###  SVD  ###
    #############
    # snr = 0
    # y, x, j_indices, noise_sigma = generator.give_batch_data_Hinput(torch.repeat_interleave(H, 1000, dim = 0), config.NT, snr_db_min=config.SNR_dBs[config.NT][snr],
    #                                                                 snr_db_max=config.SNR_dBs[config.NT][snr], 
    #                                                                 batch_size = batch_size)
    # y = y.to(device=device)

    start = time.time()
    U, singulars, V = torch.svd(H)

    Uh_real = torch.transpose(U.to(device=device), 1, 2).to(device=device)
    Vh_real = torch.transpose(V.to(device=device), 1, 2).to(device=device)

    Sigma = torch.zeros((batch_size, 2 * config.NR, 2 * config.NR))
    for ii in range(batch_size):
        Sigma[ii,:, :] = torch.diag(singulars[ii,:])

    # Uh_real = torch.repeat_interleave(Uh_real, 1000, dim=0)
    # Vh_real = torch.repeat_interleave(Vh_real, 1000, dim=0)
    # Sigma = torch.repeat_interleave(Sigma, 1000, dim=0)
    # H = torch.repeat_interleave(H, 1000, dim = 0)
    # singulars = torch.repeat_interleave(singulars, 1000, dim = 0)
    #########################################################
    ## Main loop ## 
    #########################################################

    for snr in range(0, len(config.SNR_dBs[config.NT])):
        print(config.SNR_dBs[config.NT][snr])
        # Create variables to save each trajectory and for each snr
        dist = torch.zeros((batch_size,config.n_traj))
        list_traj = torch.zeros((batch_size, 2*config.NT, config.n_traj))
        # list_traj_iter = torch.zeros((config.n_sigma_gaussian, config.n_sample_init, batch_size ,2*config.NT, config.n_traj))
        # dist_iter = torch.zeros((batch_size,config.n_sigma_gaussian * config.n_sample_init ,config.n_traj))
        # x_hat_iter = torch.zeros((batch_size, config.n_sigma_gaussian * config.n_sample_init ,2*config.NT))
        x_hat = torch.zeros((batch_size, 2*config.NT))
        ########################
        #  Samples generation  #
        ########################
        #Generate data
        # H, y, x, j_indices, noise_sigma = generator.give_batch_data(config.NT,  snr_db_min=config.SNR_dBs[config.NT][snr],
                                                                    #  snr_db_max=config.SNR_dBs[config.NT][snr], 
                                                                    #    batch_size=batch_size)        
        y, x, j_indices, noise_sigma = generator.give_batch_data_Hinput(H, config.NT, snr_db_min=config.SNR_dBs[config.NT][snr],
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
            sample_last, samples = langevin.forward(singulars.float(), Sigma.to(device=device).float(), 
                                    Uh_real.float(), Vh_real.float(), y, noise_sigma[0], batch_size, 
                                    config.NR, config.M)
            
            #Generate n_traj realizations of Langevin and then choose the best one w.r.t to ||y-Hx||^2
            list_traj[:,:,jj] = torch.clone(sample_last)
            dist[:, jj] = torch.norm(y.to(device=device) - batch_matvec_mul(H.to(device=device).float(),
                                                             sample_last.to(device=device)), 2, dim = 1)

            # list_traj_iter[:,:,:,:,jj] = torch.clone(torch.tensor(np.array(samples)))

            # index = 0
            # for ll in range(config.n_sigma_gaussian):
            #     for tt in range(config.n_sample_init):
                    
            #         dist_iter[:, index, jj] = torch.norm(y.to(device=device) - batch_matvec_mul(H.to(device=device).float(),
            #                                                     torch.tensor(samples[ll][tt]).to(device=device)), 2, dim = 1)                                                  
            #         index += 1
            print('Trajectory:', jj)

        end = time.time()
        print(end - start)

        #Pick the best trajectory for each user
        idx = torch.argsort(dist, dim=1, descending = False)

        for nn in range(0, batch_size):
            x_hat[nn, :] = torch.clone(list_traj[nn,:,idx[nn,0]])
        
        # idx = torch.argsort(dist_iter, dim=2, descending = False)

        # list_traj_iter = torch.reshape(list_traj_iter, (config.n_sigma_gaussian * config.n_sample_init, batch_size ,2*config.NT, config.n_traj))
        # for nn in range(0, batch_size):
        #     for tt in range(config.n_sample_init * config.n_sigma_gaussian):
        #     # x_hat[nn, :] = torch.clone(list_traj[nn,:,idx[nn,0]])
        #         # print(list_traj_iter[tt,nn, :, idx[nn,tt,0]].shape, x_hat_iter[nn, tt, :].shape)
        #         x_hat_iter[nn, tt, :] = torch.clone(list_traj_iter[tt,nn, :, idx[nn,tt,0]])

        # SER_lang32u_iter = []   
        # # #Evaluate performance of Langevin detector
        # for tt in range(config.n_sample_init * config.n_sigma_gaussian):
        #     SER_lang32u_iter.append(1 - sym_detection(x_hat_iter[:, tt, :].to(device='cpu'), j_indices, generator.real_QAM_const, generator.imag_QAM_const))  

        SER_lang32u.append(1 - sym_detection(x_hat.to(device='cpu'), j_indices, generator.real_QAM_const, generator.imag_QAM_const))
        print(SER_lang32u)

    return SER_lang32u, samples, x
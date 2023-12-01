################################################################################
#
####                               IMPORTING
#
################################################################################

#\\Standard libraries
from os import sched_get_priority_min
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



################################################################################
#
####                               MAIN RUN
#
################################################################################

def runClassicDetectors(config, generator, batch_size, device, H = None):

    #Define list to save data
    SER_ML = []
    SER_MMSE32u = []

    #########################################################
    ## Main loop ## 
    #########################################################
    for snr in range(0, len(config.SNR_dBs[config.NT])):
        print(config.SNR_dBs[config.NT][snr])


        ########################
        #  Samples generation  #
        ########################
        #Generate data
        # H, y, x, j_indices, noise_sigma = generator.give_batch_data(config.NT,  snr_db_min=config.SNR_dBs[config.NT][snr],
        #                                                              snr_db_max=config.SNR_dBs[config.NT][snr], 
        #                                                                batch_size=batch_size)        
        y, x, j_indices, noise_sigma = generator.give_batch_data_Hinput(H, config.NT, snr_db_min=config.SNR_dBs[config.NT][snr],
                                                                     snr_db_max=config.SNR_dBs[config.NT][snr], 
                                                                     batch_size = batch_size)
        y = y.to(device=device)
        

        #Perturbe H
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
        ########################
        ##  MMSE detector  ##
        ########################

        x_MMSE = mmse(y.float().to(device=device), H.float().to(device=device), 
                        noise_sigma.float().to(device=device), device).double()

        SER_MMSE32u.append(1 - sym_detection(x_MMSE.to(device='cpu'), j_indices, generator.real_QAM_const, 
                            generator.imag_QAM_const))
        
        
        ###############################
        ##  SDR  ##
        ###############################


        # x_sdr = blast_eval(y.unsqueeze(dim=-1).cpu().detach().numpy(), H.cpu().detach().numpy(), 
        #                 config.sigConst, config.NT, config.NR).squeeze()

        # x_sdr = sdrSolver(H.cpu().detach().numpy(), y.unsqueeze(dim=-1).cpu().detach().numpy(), 
        #                 config.sigConst, config.NT).squeeze()

        # SER_SDR.append(1 - sym_detection(torch.from_numpy(x_sdr), 
        #             j_indices, generator.real_QAM_const, generator.imag_QAM_const))

        # pool = ThreadPool(40) 
        # x_ml = pool.map(ml_proc_star, zip(H.cpu().detach().numpy(), y.unsqueeze(dim=-1).cpu().detach().numpy()))
        # x_ml = np.array(x_ml).squeeze(axis=1)
        # SER_ML.append(1 - sym_detection(torch.from_numpy(x_ml), 
        #                     j_indices, generator.real_QAM_const, generator.imag_QAM_const))

        print(SER_MMSE32u, SER_ML)


    return SER_MMSE32u, SER_ML
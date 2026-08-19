#!/usr/bin/env python

import os
import sys
import tqdm
import pdb
import importlib
import glob

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

import numpy as np
import torch
from scipy import interpolate
from skimage import color
from torch import nn

from scipy.sparse import linalg
from scipy import io
from scipy import ndimage

import torchvision
import matplotlib.pyplot as plt
import cv2

sys.path.append('modules')

import losses
import utils
import spectral
import patterns
import deep_prior
import wire
import wire2d

utils = importlib.reload(utils)
spectral = importlib.reload(spectral)
patterns = importlib.reload(patterns)
losses = importlib.reload(losses)

if __name__ == '__main__':
    expname = 'Image_Leaf1'
    exproot = 'Leaf'
    root = 'data'
        
    # Deep prior?
    rec_method = 'lowrank'
    rec_rank = 3            # If choosing lowrank
    init_nconv = 64
    num_channels_up = 4
    nettype = 'dip'

    # Learning constants
    epochs = 1000
    learning_rate = 1e-2
    lambda_tv = 3e-1
    lambda_spec = 5e-1
    specnorm_mode = 'l2'

    # For WIRE/2D WIRE
    hidden_layers = 2
    hidden_features = 160
    omega0 = 10
    sigma0 = 5

    # Load calibration data
    calib_data = io.loadmat('%s/%s/calibration.mat'%(root, exproot))
    measurement_op = calib_data['measurement_op'].astype(np.float32)[:, :-1, ...]
    wavelengths = calib_data['wavelengths'].astype(np.float32).flatten()

    [H, W, nwvl] = measurement_op.shape

    # Load measured hyperspectral cube
    rot_angle = calib_data['rot_angle'].item()
    ds_factor = calib_data['spatial_ds_factor'].item()
    crop_size = calib_data['crop_size'].flatten()

    # Load the raw image
    im = cv2.imread('%s/%s/%s.PNG'%(root, exproot, expname))[:, :, 0].astype(np.float32)

    # Rotate, crop, downsample
    imr = ndimage.rotate(im, rot_angle)
    imc = imr[crop_size[1]-1:crop_size[3], crop_size[0]-1:crop_size[2]]

    measurements = cv2.resize(imc, None, fx=1/ds_factor, 
                                fy=1/ds_factor, interpolation=cv2.INTER_AREA)
    
    measurements = utils.normalize(measurements, True)

    # Now generate variables
    if rec_method == 'tv':
        hsten_raw = torch.rand(1, H, W, nwvl).cuda()/10.0
        hsten_raw = torch.autograd.Variable(hsten_raw, requires_grad=True).cuda()
        hsten_raw = torch.nn.Parameter(hsten_raw)
        
        params = [hsten_raw]
    elif rec_method == 'dip':
        hs_net = deep_prior.get_net(8, 'skip', 'reflection2d',
                                    upsample_mode='bilinear',
                                    skip_n33d=init_nconv,
                                    skip_n33u=init_nconv,
                                    num_scales=num_channels_up,
                                    n_channels=nwvl).cuda()
        hs_inp = deep_prior.get_noise(8, 'noise', [H, W]).cuda().detach()

        with torch.no_grad():
            hs_inp = hs_inp/10.0
                
        params = hs_net.parameters()
        
        lambda_tv = 0

    elif rec_method == 'lowrank':
        im_net = deep_prior.get_net(8, 'skip', 'reflection2d',
                                    upsample_mode='bilinear',
                                    skip_n33d=init_nconv,
                                    skip_n33u=init_nconv,
                                    num_scales=num_channels_up,
                                    n_channels=rec_rank).cuda()
        im_inp = deep_prior.get_noise(8, 'noise', [H, W]).cuda().detach()

        spec_raw = torch.rand(1, nwvl, rec_rank).cuda()/10.0
        spec_raw = torch.autograd.Variable(spec_raw, requires_grad=True).cuda()
        spec_raw = torch.nn.Parameter(spec_raw)

        params = list(im_net.parameters()) + [spec_raw]

        lambda_tv = 0

    elif rec_method == 'wire' or rec_method =='wire2d':
        x = torch.linspace(-1, 1, W)
        y = torch.linspace(-1, 1, H)
        
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        model_input = torch.cat((X[None, :, :, None],
                                 Y[None, :, :, None]),
                                dim=-1).cuda()
        
        if rec_method == 'wire':
            INR = wire.INR
        else:
            INR = wire2d.INR
        model = INR(in_features=2,
                         hidden_features=hidden_features,
                         hidden_layers=hidden_layers,
                         out_features=nwvl,
                         first_omega_0=omega0,
                         hidden_omega_0=omega0,
                         scale=sigma0).cuda()
        params = model.parameters()

        lambda_tv = 0

    elif rec_method == 'wire_lowrank':
        x = torch.linspace(-1, 1, W)
        y = torch.linspace(-1, 1, H)
        
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        model_input = torch.cat((X[None, :, :, None],
                                 Y[None, :, :, None]),
                                dim=-1).cuda()
        
        if rec_method == 'wire':
            INR = wire.INR
        else:
            INR = wire2d.INR
        model = INR(in_features=2,
                         hidden_features=hidden_features,
                         hidden_layers=hidden_layers,
                         out_features=rec_rank,
                         first_omega_0=omega0,
                         hidden_omega_0=omega0,
                         scale=sigma0).cuda()
        params = model.parameters()

    measurements_ten = torch.tensor(measurements)[None, ...].cuda()
    meas_op_ten = torch.tensor(measurement_op)[None, ...].cuda()
    
    # Create optimizer
    
    optimizer = torch.optim.Adam(lr=learning_rate, params=params)
    
    criterion_l1 = losses.L2Norm()
    criterion_tv = losses.TVNorm()
    criterion_spec = losses.SpecNorm(mode=specnorm_mode)
    
    loss_array = np.zeros(epochs)
    mse_array = np.zeros(epochs)
    
    best_loss = float('inf')
    best_epoch = 0
    
    tbar = tqdm.tqdm(range(epochs))
    for idx in tbar:  
        if rec_method == 'dip':
            hs_estim = hs_net(hs_inp)
            hs_estim = hs_estim.permute(0, 2, 3, 1)
        elif rec_method == 'lowrank':
            im_estim = im_net(im_inp)

            hs_ten = torch.bmm(spec_raw, im_estim.reshape(1, rec_rank, -1))
            hs_estim = hs_ten.reshape(1, nwvl, H, W).permute(0, 2, 3, 1)
        elif rec_method == 'wire' or rec_method == 'wire2d':
            hs_estim = model(model_input)
        else:
            hs_estim = hsten_raw
            
        measurement_estim = (hs_estim*meas_op_ten).sum(3)
            
        loss_l1 = criterion_l1(measurements_ten - measurement_estim)
        loss_tv = criterion_tv(hs_estim.permute(0, 3, 1, 2))

        if rec_method == 'lowrank':
            loss_spec = criterion_spec(spec_raw)
        else:
            loss_spec = criterion_spec(hs_estim.permute(0, 3, 1, 2))
        
        loss = loss_l1 + lambda_tv*loss_tv + lambda_spec*loss_spec
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_array[idx] = loss.item()
        
        if loss_array[idx] < best_loss:
            best_loss = loss_array[idx]
            best_epoch = idx
            best_hsten = np.copy(hs_estim.detach().cpu().squeeze().numpy())
        
        tbar.set_description('Loss: %.2e'%loss_array[idx])
        tbar.refresh()
        
        with torch.no_grad():
            img = hs_estim[0, ..., idx%nwvl].detach().cpu().numpy()
            
            cv2.imshow('Rec', utils.normalize(img, True))
            cv2.waitKey(1)
            
    # Compute metrics
    hypercube_rec = best_hsten.astype(np.float32)
    
    imrgb_rec = spectral.hyper2rgb(hypercube_rec, wavelengths)
        
    mdict = {'rec': hypercube_rec,
             'wvl': wavelengths,
             'measurements': measurements,
             'filter_cube': measurement_op}    

    os.makedirs('results/%s'%exproot, exist_ok=True)
    io.savemat('results/%s/%s_%s.mat'%(exproot, expname, rec_method), mdict)

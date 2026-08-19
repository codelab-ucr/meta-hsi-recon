#!/usr/bin/env python

import os
import sys
import tqdm
import pdb

import numpy as np
import torch
from scipy import interpolate
from skimage import color
from torch import nn

from scipy import linalg
import scipy.sparse.linalg as sp_linalg

import torchvision
import cv2

import losses

def downsample_wvl(hypercube, src_wvl, dst_wvl):
    '''
        Downsample a hypercube along its wavelength axis using anti-aliased,
        box-based interpolation.

        Inputs:
            hypercube: (H, W, nwvl) hypercube
            src_wvl: Source wavelengths
            dst_wvl: Destination wavelengths

        Outputs:
            hypercube_ds: Downsampled hypercube
    '''

    H, W, nwvl_src = hypercube.shape
    nwvl_dst = dst_wvl.size

    hypercube_ds = np.zeros((H, W, nwvl_dst), dtype=hypercube.dtype)

    # Assuming linear wavelength spacing
    dwvl = abs(dst_wvl[1] - dst_wvl[0])

    for idx in range(nwvl_dst):
        mask = (src_wvl >= dst_wvl[idx]-dwvl/2)*(src_wvl < dst_wvl[idx]+dwvl/2)
        hypercube_ds[..., idx] = (hypercube*mask.reshape(1, 1, nwvl_src)).sum(2)/mask.sum()

    return hypercube_ds


def guided_filter(hypercube, imrgb, radius=5, eps=1e-5):
    '''
        Guide filter a hyperspectral image
    '''
    gf = cv2.ximgproc.createGuidedFilter(imrgb.mean(2).astype(np.float32),
                                          radius, eps)
        
    H, W, nwvl = hypercube.shape
    hypercube_filtered = np.zeros_like(hypercube)
    for idx in range(nwvl):
        img = hypercube[..., idx]
        hypercube_filtered[..., idx] = gf.filter(img)
        
    return hypercube_filtered

def uv_decompose(cube, rank=6, learning_rate=1e-2, epochs=2000, gt=None,
                  lambda_spec=1e-1, lambda_tv=1e-2):
    '''
        Reconstruct a hyperspectral cube using simple convex optimization
        approach with low-rank matrix factorization
        
        Inputs:
            cube: Input cube for decomposing
            rank: Rank for matrix factorization
            learning_rate: Learning rate for SGD
            epochs: Iterations for learning
            gt: Ground truth hypercube, or None
            lambda_spec: Weight for spectral smoothness loss
            lambda_tv: Weight for TV loss on space
            
        Outputs:
            cube_rec: Recovered cube
            U_estim, V_estim: Matrix factorization   
            loss_array: if hypercube is provided, return loss per iteration  
    '''
    H, W, nwvl = cube.shape
    
    # Convert to pytorch tensor
    if gt is not None:
        gtten = torch.tensor(gt.reshape(H*W, nwvl)).cuda()[None, ...]
        imref = gt.mean(2)
        
    inp = torch.tensor(cube.reshape(H*W, nwvl)).cuda()[None, ...]
    
    # Create U and V variables
    U = torch.rand(1, H*W, rank)
    V = torch.rand(1, rank, nwvl)
    
    Uvar = torch.autograd.Variable(U, requires_grad=True).cuda()
    Uparam = torch.nn.Parameter(Uvar)
    
    Vvar = torch.autograd.Variable(V, requires_grad=True).cuda()
    Vparam = torch.nn.Parameter(Vvar)
    
    # Create optimizer
    optimizer = torch.optim.Adam(lr=learning_rate,
                                 params=[Uparam]+[Vparam])
    
    loss_array = np.zeros(epochs)
    iterbar = tqdm.trange(epochs, desc='Epochs', leave=True)
    
    criterion_tv = losses.TVNorm()
    for epoch in iterbar:
        hsten_estim = torch.bmm(Uparam, Vparam)
        
        loss1 = abs(hsten_estim - inp).mean()
        
        Uimg = Uparam.reshape(1, H, W, rank).permute(3, 0, 1, 2)
        loss2 = criterion_tv(Uimg)
        loss3 = ((Vparam[:, :, 1:]-Vparam[:, :, :-1])**2).mean()
        
        loss = loss1 + lambda_tv*loss2 + lambda_spec*loss3
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_array[epoch] = loss.item()
        iterbar.set_description('Loss: %e'%loss_array[epoch])
        iterbar.refresh()
        
        if gt is not None:
            imrec = hsten_estim.mean(2).detach().cpu().numpy().squeeze()
            with torch.no_grad():
                imdiff = abs(hsten_estim - gtten).mean(-1)[0, ...]
                imdiff = imdiff.detach().cpu().numpy()
            cv2.imshow('Recon', np.hstack((imref, imrec.reshape(H, W))))
            cv2.imshow('Diff', imdiff.reshape(H, W)*10)
            cv2.waitKey(1)
        
    # Detach variables and reconstruct
    U_estim = Uparam.detach().cpu().numpy()[0, ...]
    V_estim = Vparam.detach().cpu().numpy()[0, ...]
    
    hsmat_estim = hsten_estim.detach().cpu().numpy()[0, ...]
    
    cube_rec = hsmat_estim.reshape(H, W, nwvl)
    
    return cube_rec, U_estim, V_estim

def lr_decompose(cube, rank):
    '''
        Perform a truncated SVD
        
        Inputs:
            cube: (H, W, nwvl) hyperspectral cube
            rank: Rank to decompose the cube
            
        Outputs:
            cube_lr: Low rank decomposition
    '''
    H, W, nwvl = cube.shape
    hsmat = cube.reshape(H*W, nwvl)
    
    u, s, vt = sp_linalg.svds(hsmat, k=rank)
    
    hsmat_lr = u.dot(np.diag(s)).dot(vt)
    cube_lr = hsmat_lr.reshape(H, W, nwvl)
    
    return cube_lr

def hyper2xyz(imhyper, wavelengths):
    '''
        Function to convert a hyperspectral image to XYZ image.

        Inputs:
            imhyper: 3D Hyperspectral image.
            wavelengths: Wavelengths corresponding to each slice.
            gamma: Gamma correction constant. Default is 1.

        Outputs:
            imxyz: XYZ image.
    '''
    cmf_data = np.genfromtxt('modules/lin2012xyz2e_1_7sf.csv', delimiter=',')

    # Interpolate the wavelengths and x, y, z values
    cmf_data_new = np.zeros((len(wavelengths), 3))
    for idx in range(3):
        interp_func = interpolate.interp1d(cmf_data[:, 0],
                                           cmf_data[:, idx+1],
                                           kind='linear',
                                           fill_value='extrapolate')
        cmf_data_new[:, idx] = interp_func(wavelengths)

    # Find valid indices for converting to RGB image.
    #valid_idx = np.where((wavelengths > min(l)) & (wavelengths < max(l)))

    [H, W, T] = imhyper.shape
    hypermat = imhyper.reshape(H*W, T)

    # Compute XYZ image
    imxyz = np.dot(hypermat, cmf_data_new);
    imxyz = imxyz.reshape(H, W, 3)
    
    return imxyz
    
def hyper2rgb(imhyper, wavelengths, gamma=1, normalize=True):
    '''
        Function to convert a hyperspectral image to RGB image.

        Inputs:
            imhyper: 3D Hyperspectral image.
            wavelengths: Wavelengths corresponding to each slice.
            gamma: Gamma correction constant. Default is 1.

        Outputs:
            imrgb: RGB image.
    '''
    imxyz = hyper2xyz(imhyper, wavelengths)

    # Before you convert to rgb, normalize
    if normalize:
        imxyz /= imxyz.max()

    # Compute RGB image from xyz
    imrgb = pow(color.xyz2rgb(imxyz), 1.0/gamma)

    return imrgb

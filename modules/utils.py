#!/usr/bin/env python

'''
    Miscellaneous utilities that are extremely helpful but cannot be clubbed
    into other modules.
'''

# System imports
import os
import sys
import time
import pickle
import pdb
import glob

# Scientific computing
import numpy as np
import scipy as sp
import scipy.linalg as lin
import scipy.ndimage as ndim
from scipy import io
from scipy.sparse.linalg import svds
from scipy import signal
from scipy import interpolate

from skimage.metrics import structural_similarity as ssim_func

# Plotting
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def implay(cube, delay=20):
    '''
        Play hyperspectral image as a video
    '''
    if cube.dtype != np.uint8:
        cube = (255*cube/cube.max()).astype(np.uint8)
    
    T = cube.shape[-1]
    
    for idx in range(T):
        cv2.imshow('Video', cube[..., idx])
        cv2.waitKey(delay)

def interp1(data, x_data, xq):
    '''
        Interpolate each row of data for desired wavelengths
    '''
    ndata, nx = data.shape
    
    output = np.zeros((ndata, xq.size))
    
    for idx in range(ndata):
        y = data[idx, ...]
        func = interpolate.interp1d(x_data, y, kind='linear',
                                    fill_value=0, bounds_error=False)
        output[idx, ...] = func(xq)
        
    return output

def load_raw(filename, nrows, ncols, dtype=np.uint16):
    '''
        Load a RAW image from a file

        Inputs:
            filename: Name of the file to load. Should have an
                extension of '.Raw'
            nrows, ncols: Number of rows and columns in the image
            dtype: Datatype of the file

        Outputs:
            im: Loaded image
    '''
    with open(filename, 'rb') as fd:
        rawdata = np.fromfile(fd, dtype=dtype, count=nrows*ncols)

    im = rawdata.reshape(nrows, ncols)

    return im

def stack2mosaic(imstack):
    '''
        Convert a 3D stack of images to a 2D mosaic

        Inputs:
            imstack: (H, W, nimg) stack of images

        Outputs:
            immosaic: A 2D mosaic of images
    '''
    H, W, nimg = imstack.shape

    nrows = int(np.ceil(np.sqrt(nimg)))
    ncols = int(np.ceil(nimg/nrows))

    immosaic = np.zeros((H*nrows, W*ncols), dtype=imstack.dtype)

    for row_idx in range(nrows):
        for col_idx in range(ncols):
            img_idx = row_idx*ncols + col_idx
            if img_idx >= nimg:
                return immosaic

            immosaic[row_idx*H:(row_idx+1)*H, col_idx*W:(col_idx+1)*W] = \
                                              imstack[:, :, img_idx]

    return immosaic

def nextpow2(x):
    '''
        Return smallest number larger than x and a power of 2.
    '''
    logx = np.ceil(np.log2(x))
    return pow(2, logx)

def normalize(x, fullnormalize=False):
    '''
        Normalize input to lie between 0, 1.

        Inputs:
            x: Input signal
            fullnormalize: If True, normalize such that minimum is 0 and
                maximum is 1. Else, normalize such that maximum is 1 alone.

        Outputs:
            xnormalized: Normalized x.
    '''

    if x.sum() == 0:
        return x
    
    xmax = x.max()

    if fullnormalize:
        xmin = x.min()
    else:
        xmin = 0

    xnormalized = (x - xmin)/(xmax - xmin)

    return xnormalized

def asnr(x, xhat, compute_psnr=False):
    '''
        Compute affine SNR, which accounts for any scaling and shift between two
        signals

        Inputs:
            x: Ground truth signal(ndarray)
            xhat: Approximation of x

        Outputs:
            asnr_val: 20log10(||x||/||x - (a.xhat + b)||)
                where a, b are scalars that miminize MSE between x and xhat
    '''
    mxy = (x*xhat).mean()
    mxx = (xhat*xhat).mean()
    mx = xhat.mean()
    my = x.mean()
    

    a = (mxy - mx*my)/(mxx - mx*mx)
    b = my - a*mx

    if compute_psnr:
        return psnr(x, a*xhat + b)
    else:
        return rsnr(x, a*xhat + b)

def rsnr(x, xhat):
    '''
        Compute reconstruction SNR for a given signal and its reconstruction.

        Inputs:
            x: Ground truth signal (ndarray)
            xhat: Approximation of x

        Outputs:
            rsnr_val: RSNR = 20log10(||x||/||x-xhat||)
    '''
    xn = lin.norm(x.reshape(-1))
    en = lin.norm((x-xhat).reshape(-1)) + 1e-12
    rsnr_val = 20*np.log10(xn/en)

    return rsnr_val

def psnr(x, xhat):
    ''' Compute Peak Signal to Noise Ratio in dB

        Inputs:
            x: Ground truth signal
            xhat: Reconstructed signal

        Outputs:
            snrval: PSNR in dB
    '''
    err = x - xhat
    denom = np.mean(pow(err, 2)) + 1e-12

    snrval = 10*np.log10((np.max(x)**2)/denom)

    return snrval

def ssim(x, xhat):
    if x.ndim == 2:
        return ssim_func(x, xhat)
    else:
        return ssim_func(x, xhat, multichannel=True)

def SAM_3d(x, xhat, avg=False):
    '''
        Compute SAM for a 3D hyperspectral cube

        Inputs:
            x: Ground truth HSI
            xhat: Reconstructed HSI
            avg: If True, average spatially

        Outputs:
            SAM: SAM map (or average value)
    '''

    x_norm = (x*x).sum(2)
    xhat_norm = (xhat*xhat).sum(2)

    xxhat = abs(x*xhat).sum(2)

    SAM = np.arccos(xxhat/np.sqrt(x_norm*xhat_norm + 1e-12))*180/np.pi

    if avg:
        SAM = np.mean(SAM)

    return SAM

def savep(data, filename):
    '''
        Tiny wrapper to store data as a python pickle.

        Inputs:
            data: List of data
            filename: Name of file to save
    '''
    f = open(filename, 'wb')
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def loadp(filename):
    '''
        Tiny wrapper to load data from python pickle.

        Inputs:
            filename: Name of file to load from

        Outputs:
            data: Output data from pickle file
    '''
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()

    return data

def display_time(total_time):
    '''
        Tiny wrapper to print time in an appropriate way.

        Inputs:
            total_time: Raw time in seconds

        Outputs:
            None
    '''
    if total_time < 60:
        print('Total scanning time: %.2f seconds'%total_time)
    elif total_time < 3600:
        print('Total scanning time: %.2f minutes'%(total_time/60))
    elif total_time < 86400:
        print('Total scanning time: %.2f hours'%(total_time/3600))
    else:
        print('Total scanning time: %.2f days'%(total_time/86400))
        print('... what are you really doing?')

def dither(im):
    '''
        Implements Floyd-Steinberg spatial dithering algorithm

        Inputs:
            im: Grayscale image normalized between 0, 1

        Outputs:
            imdither: Dithered image
    '''
    H, W = im.shape
    imdither = np.zeros((H+1, W+1))

    # Pad the last row/column to propagate error
    imdither[:H, :W] = im
    imdither[H, :W] = im[H-1, :W]
    imdither[:H, W] = im[:H, W-1]
    imdither[H, W] = im[H-1, W-1]

    for h in range(0, H):
        for w in range(1, W):
            oldpixel = imdither[h, w]
            newpixel = (oldpixel > 0.5)
            imdither[h, w] = newpixel

            err = oldpixel - newpixel
            imdither[h, w+1] += (err * 7.0/16)
            imdither[h+1, w-1] += (err * 3.0/16)
            imdither[h+1, w] += (err * 5.0/16)
            imdither[h+1, w+1] += (err * 1.0/16)

    return imdither[:H, :W]

def embed(im, embedsize):
    '''
        Embed a small image centrally into a larger window.

        Inputs:
            im: Image to embed
            embedsize: 2-tuple of window size

        Outputs:
            imembed: Embedded image
    '''

    Hi, Wi = im.shape
    He, We = embedsize

    dH = (He - Hi)//2
    dW = (We - Wi)//2

    imembed = np.zeros((He, We), dtype=im.dtype)
    imembed[dH:Hi+dH, dW:Wi+dW] = im

    return imembed

def measure(x, noise_snr=40, tau=100):
    ''' Realistic sensor measurement with readout and photon noise

        Inputs:
            noise_snr: Readout noise in electron count
            tau: Integration time. Poisson noise is created for x*tau.
                (Default is 100)

        Outputs:
            x_meas: x with added noise
    '''
    x_meas = np.copy(x)

    #noise = pow(10, -noise_snr/20)*np.random.randn(x_meas.size).reshape(x_meas.shape)
    noise = np.random.randn(x_meas.size).reshape(x_meas.shape)*noise_snr

    # First add photon noise, provided it is not infinity
    if tau != float('Inf'):
        x_meas = x_meas*tau

        x_meas[x > 0] = np.random.poisson(x_meas[x > 0])
        x_meas[x <= 0] = -np.random.poisson(-x_meas[x <= 0])

        x_meas = (x_meas + noise)/tau

    else:
        x_meas = x_meas + noise

    return x_meas

def deconvwnr1(sig, kernel, wconst=1e-2):
    '''
        Deconvolve a 1D signal using Wiener deconvolution

        Inputs:
            sig: Input signal
            kernel: Impulse response
            wconst: Wiener deconvolution constant

        Outputs:
            sig_deconv: Deconvolved signal
    '''

    sigshape = sig.shape
    sig = sig.ravel()
    kernel = kernel.ravel()

    N = sig.size
    M = kernel.size

    # Padd signal to regularize 
    sig_padded = np.zeros(N+2*M)
    sig_padded[M:-M] = sig

    # Compute Fourier transform
    sig_fft = np.fft.fft(sig_padded)
    kernel_fft = np.fft.fft(kernel, n=(N+2*M))

    # Compute inverse kernel
    kernel_inv_fft = np.conj(kernel_fft)/(np.abs(kernel_fft)**2 + wconst)

    # Now compute deconvolution
    sig_deconv_fft = sig_fft*kernel_inv_fft

    # Compute inverse fourier transform
    sig_deconv_padded = np.fft.ifft(sig_deconv_fft)

    # Clip
    sig_deconv = np.real(sig_deconv_padded[M//2:M//2+N])

    return sig_deconv.reshape(sigshape)

def lowpassfilter(data, order=5, freq=0.5):
    '''
        Low pass filter the input data with butterworth filter.
        This is based on Zackory's github repo: 
            https://github.com/Healthcare-Robotics/smm50

        Inputs:
            data: Data to be filtered with each row being a spectral profile
            order: Order of butterworth filter
            freq: Cutoff frequency

        Outputs:
            data_smooth: Smoothed spectral profiles
    '''
    # Get butterworth coefficients
    b, a = signal.butter(order, freq, analog=False)

    # Then just apply the filter
    data_smooth = signal.filtfilt(b, a, data)

    return data_smooth

def grid_plot(imdata):
    '''
        Plot 3D set of images into a 2D grid using subplots.

        Inputs:
            imdata: N x H x W image stack

        Outputs:
            None
    '''
    N, H, W = imdata.shape

    nrows = int(np.sqrt(N))
    ncols = int(np.ceil(N/nrows))

    for idx in range(N):
        plt.subplot(nrows, ncols, idx+1)
        plt.imshow(imdata[idx, :, :], cmap='gray')
        plt.xticks([], [])
        plt.yticks([], [])
        
def build_montage(images):
    '''
        Build a montage out of images
    '''
    nimg, H, W = images.shape
    
    nrows = int(np.ceil(np.sqrt(nimg)))
    ncols = int(np.ceil(nimg/nrows))
    
    montage_im = np.zeros((H*nrows, W*ncols), dtype=np.float32)
    
    cnt = 0
    for r in range(nrows):
        for c in range(ncols):
            h1 = r*H
            h2 = (r+1)*H
            w1 = c*W
            w2 = (c+1)*W

            if cnt == nimg:
                break

            montage_im[h1:h2, w1:w2] = images[cnt, ...]
            cnt += 1
    
    return montage_im
      
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

def ims2rgb(im1, im2):
    '''
        Concatenate images into RGB
        
        Inputs:
            im1, im2: Two images to compare
    '''
    H, W = im1.shape
    
    imrgb = np.zeros((H, W, 3))
    imrgb[..., 0] = im1
    imrgb[..., 2] = im2

    return imrgb

def textfunc(im, txt):
    return cv2.putText(im, txt, (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (1, 1, 1),
                        2,
                        cv2.LINE_AA)

def boxify(im, topleft, boxsize, color=[1, 1, 1], width=2):
    '''
        Generate a box around a region.
    '''
    h, w = topleft
    dh, dw = boxsize
    
    im[h:h+dh+1, w:w+width, :] = color
    im[h:h+width, w:w+dh+width, :] = color
    im[h:h+dh+1, w+dw:w+dw+width, :] = color
    im[h+dh:h+dh+width, w:w+dh+width, :] = color

    return im

def resize(cube, scale):
    '''
        Resize a multi-channel image
        
        Inputs:
            cube: (H, W, nchan) image stack
            scale: Scaling 
    '''
    H, W, nchan = cube.shape
    
    im0_lr = cv2.resize(cube[..., 0], None, fx=scale, fy=scale)
    Hl, Wl = im0_lr.shape
    
    cube_lr = np.zeros((Hl, Wl, nchan), dtype=cube.dtype)
    
    for idx in range(nchan):
        cube_lr[..., idx] = cv2.resize(cube[..., idx], None,
                                       fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
    return cube_lr

def crop_center_array(arr, crop_rows, crop_cols):
    """
    Center-crop a numpy array image.

    Args:
        arr: Numpy array of size H x W x () to be cropped.
        crop_nrows: Number of rows in crop.
        crop_ncols: Number of cols in crop.

    Returns:
        Numpy array of size crop_nrows x crop_ncols x ()
    """
    nrows = arr.shape[0]
    ncols = arr.shape[1]
    start_row = nrows // 2 - crop_rows // 2
    start_col = ncols // 2 - crop_cols // 2
    return arr[start_row:start_row+crop_rows, start_col:start_col+crop_cols]

def loadmat(filepath):
    arrays = dict()
    f = h5py.File(filepath)
    
    for k, v in f.items():
        arrays[k] = np.array(v)
        
    return arrays
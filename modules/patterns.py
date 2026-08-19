#!/usr/bin/env python

import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
plt.gray()


def generate_snake_tile(siz = np.array([16, 16], dtype=np.int16)):
	amat = np.zeros( siz, dtype=np.int16)
	loc = np.array([0, 0], dtype=np.int16) 
	val = 0
	step = np.array([-1, 1], dtype=np.int16)

	amat[loc[0], loc[1]] = val
	
	for xx in range(np.prod(siz)-1):
		loc = loc + step
		change_dir = False
		if step[0] == -1:
			flag1 = (loc[0] == -1)
			flag2 = (loc[1] == siz[1])
			if (flag1 & flag2):
				loc[0] = 1
				loc[1] = siz[1]-1
				change_dir = True
			else:
				if flag1:
					loc[0] = 0
					change_dir = True
				if flag2:
					loc[0] = loc[0] + 2
					loc[1] = siz[1] - 1
					change_dir = True
		
			
	
		if step[0] == 1:
			flag1 = (loc[1] == -1)
			flag2 = (loc[0] == siz[0])
			if (flag1 & flag2):
				loc[1] = 1
				loc[0] = siz[0]-1
				change_dir = True
			else:
				if flag1:
					loc[1] = 0
					change_dir = True
				if flag2:
					loc[1] = loc[1] + 2
					loc[0] = siz[0] - 1
					change_dir = True
		val = val + 1 
		amat[loc[0], loc[1]] = val

		if change_dir:
			step = -step	


	return amat

def generate_diagonal_tile(siz = np.array([16, 16], dtype=np.int16)):
	amat = np.zeros( siz, dtype=np.int16)
	loc = np.array([0, 0], dtype=np.int16) 
	val = 0
	step = np.array([-1, 1], dtype=np.int16)

	amat[loc[0], loc[1]] = val
	
	col_idx = 0
	row_idx = 0
	
	for xx in range(np.prod(siz)-1):
		loc = loc + step
		
		flag1 = (loc[0] == -1)
		flag2 = (loc[1] == siz[1])
		
		if (flag1 & flag2):
			loc[0] = siz[0]-1
			loc[1] = 1
			row_idx = 1
		else:
			if flag1:
				col_idx = col_idx + 1
				loc[0] = col_idx
				loc[1] = 0				
			if flag2:
				row_idx = row_idx + 1
				loc[0] = siz[0] - 1
				loc[1] = row_idx
			
	
		val = val + 1 
		amat[loc[0], loc[1]] = val

	return amat
	
def generate_horizontal_tile(siz = np.array([16, 16], dtype=np.int16)):
	x = np.linspace(0, siz[0]-1, siz[0])
	y = np.linspace(0, siz[1]-1, siz[1])
	X, Y = np.meshgrid(x, y)
	amat = X + Y*siz[0]
	return amat

def generate_vertical_tile(siz = np.array([16, 16], dtype=np.int16)):
	x = np.linspace(0, siz[0]-1, siz[0])
	y = np.linspace(0, siz[1]-1, siz[1])
	X, Y = np.meshgrid(x, y)
	amat = Y + X*siz[1]
	return amat
	
def generate_random_pattern(siz = np.array([16, 16], dtype=np.int16), out_size = np.array([32, 32], dtype=np.int16)):
	
	rnd1 = np.random.rand(out_size[0], out_size[1])
	rnd2 = gaussian_filter(rnd1, np.double(siz[0])/2)
	rnd2 = (rnd2 - np.amin(rnd2))/(np.amax(rnd2)-np.amin(rnd2))
	
	amat = (255*rnd2).astype(np.uint8)
	
	return amat
	
def speckle(pattern_size, freq_contour):
	'''
		Generate random speckle pattern instead of gaussian filtered pattern
		
		Inputs:
			pattern_size: Size of the SLM
			freq_contour: Contour size, as defined in:
				https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?referer=https://www.researchgate.net/&httpsredir=1&article=1119&context=ece_fac
	
		Outputs:
			speckle_im: Random speckle pattern
	'''
	H, W = pattern_size
	[Y, X] = np.mgrid[:H, :W]
	
	mask = np.hypot((X - W/2), (Y - H/2)) < freq_contour/2
	
	speckle_im_fft_phase = np.exp(1j*(-np.pi + 2*np.pi*np.random.rand(H, W)))
	speckle_im_fft = speckle_im_fft_phase*mask.astype(np.float32)
	
	speckle_im = abs(np.fft.fft2(speckle_im_fft))**2
	speckle_im /= speckle_im.max()
					
	return (255*speckle_im).astype(np.uint8)
	
def generate_tile(siz = np.array([16, 16], dtype=np.int16), mirrorFlip = True, TileType = "H", int_rep = 1, out_size = np.array([8,8], dtype=np.int16), boundary = False):
	
	if TileType == "H":
		amat = generate_horizontal_tile(siz)
	if TileType == "V":
		amat = generate_vertical_tile(siz)
	if TileType == "D":
		amat = generate_diagonal_tile(siz)
	if TileType == "S":
		amat = generate_snake_tile(siz)
	if TileType == "R":
		amat = generate_random_pattern(siz, out_size)
		return amat
	
	
	if boundary:
		bmat = np.zeros((np.size(amat, 0)+2, np.size(amat, 1)+2))
		bmat[0,0] = amat[0,0]
		bmat[-1, 0] = amat[-1, 0]
		bmat[-1, -1] = amat[-1, -1]
		bmat[0, -1] = amat[0, -1]
		bmat[1:-1,0] = amat[:, 0]
		bmat[1:-1,-1] = amat[:, -1]
		bmat[0, 1:-1] = amat[0, :]
		bmat[-1, 1:-1] = amat[-1, :]
		bmat[1:-1, 1:-1] = amat
		amat = bmat
		siz = np.size(bmat)
	
	if mirrorFlip:
		amat = np.hstack((amat, amat[:, ::-1]))
		amat = np.vstack((amat, amat[::-1, :]))
		
	
		
	amat = np.kron( amat, np.ones([int_rep, int_rep], dtype=np.int16))
	repnum = np.ceil(np.divide(out_size, amat.shape, dtype=np.double))
	amat = np.tile(amat, np.array(repnum, dtype=np.int))
	amat = amat[:out_size[0], :out_size[1]]
	return amat
	
def get_pattern_defunct(pattern_size, pattern_type='snake', tile_size=[16, 16]):
	'''
		Get pattern for spectral modulation.
		
		Inputs:
			pattern_size: Size of the full pattern
			pattern_type: one of snake, diagonal, horizontal, vertical, 
				random, or speckle
			tile_size: If snake, diagonal, horizontal or vertical, tile_size
				dictates the size of smallest repetitive tile
		Outputs:
			pattern: Modulation pattern
	'''
	if pattern_type == 'snake':
		tile = generate_snake_tile(np.array(tile_size))
	elif pattern_type == 'diagonal':
		tile = generate_diagonal_tile(np.array(tile_size))
	elif pattern_type == 'horizontal':
		tile = generate_horizontal_tile(np.array(tile_size))
	elif pattern_type == 'vertical':
		tile = generate_vertical_tile(np.array(tile_size))
	else:
		if pattern_type == 'random':
			pattern = generate_random_pattern(siz=np.array(tile_size),
											out_size=pattern_size)
		else:
			freq_contour = max(pattern_size[0]/tile_size[0],
							pattern_size[1]/tile_size[1])
			pattern = speckle(pattern_size, 2*freq_contour)
		return pattern
	
	pattern = np.tile(tile, [pattern_size[0]//tile_size[0] + 1,
							pattern_size[1]//tile_size[1] + 1])
	pattern = pattern[:pattern_size[0], :pattern_size[1]]
	return pattern

def get_pattern(pattern_size, pat_idx, root='../../capture_code/patterns'):
	pat_sub = (plt.imread('%s/%.2d.png'%(root, pat_idx+1))*255).astype(np.uint8)
	pat_sub = np.arange(256).reshape(16, 16).astype(np.float32)
	tile_size = pat_sub.shape
	pattern = np.tile(pat_sub, [pattern_size[0]//tile_size[0] + 1,
								pattern_size[1]//tile_size[1] + 1])
	pattern = pattern[:pattern_size[0], :pattern_size[1]]
	return pattern.astype(np.int16)
    

if __name__ == '__main__':
	pattern = get_pattern([1080, 1920], 'speckle')
	plt.imshow(pattern)
	plt.show()

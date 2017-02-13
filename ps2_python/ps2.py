# ps2
import os
import sys
import numpy as np
import cv2
from disparity_ssd import *
from disparity_ncorr import *

use_subsampling = True

## 1: SSD match algorithm
def ps2_1():
    # Read images
    L = cv2.imread('input/pair0-L.png', cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)
    R = cv2.imread('input/pair0-R.png', cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)
    # Compute disparity using method ssd template matching
    D_L = disparity_ssd(L, R, block_size=11, disparity_range=30)
    D_R = disparity_ssd(R, L, block_size=11, disparity_range=30)
    # shift and scale disparity maps
    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # save disparity maps
    cv2.imwrite('output/ps2-1-a-1.png', D_L)
    cv2.imwrite('output/ps2-1-a-2.png', D_R)

# 2: Disparity map mathcing on a real image pair
def ps2_2():
    L = cv2.imread('input/pair1-L.png', cv2.IMREAD_GRAYSCALE)
    R = cv2.imread('input/pair1-R.png', cv2.IMREAD_GRAYSCALE)
    #  subsample and convert to [0,1] scale
    if use_subsampling:
        L = cv2.pyrDown(L)
        R = cv2.pyrDown(R)
    L = L * (1.0 / 255.0)
    R = R * (1.0 / 255.0)
    # compute disparity maps
    bs = 7; dr = 100; l = 0.0
    D_L = np.abs(disparity_ssd(L, R, block_size=bs, disparity_range=dr, lambda_factor=l))
    D_R = np.abs(disparity_ssd(R, L, block_size=bs, disparity_range=dr, lambda_factor=l))
    # shift and scale disparity maps
    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #  revert dimensions by upsampling if previously downsampled
    if use_subsampling:
        D_L = cv2.pyrUp(D_L)
        D_R = cv2.pyrUp(D_R)
    cv2.imwrite('output/ps2-2-a-1.png', D_L)
    cv2.imwrite('output/ps2-2-a-2.png', D_R)

# 3: Demonstration of the sensitivity of SSD to perturbations
def ps2_3():
    # read real image pair in grayscale
    L = cv2.imread('input/pair1-L.png', cv2.IMREAD_GRAYSCALE)
    R = cv2.imread('input/pair1-R.png', cv2.IMREAD_GRAYSCALE)
    #  subsample and convert to [0,1] scale
    if use_subsampling:
        L = cv2.pyrDown(L)
        R = cv2.pyrDown(R)
    L = L * (1.0 / 255.0)
    R = R * (1.0 / 255.0)
    # add noise or increase contrast to the left image
    noise = np.empty(L.shape, np.float32)
    cv2.randn(noise, 0, 0.05);
    L_n = L + noise
    L_c = L * 1.1
    # compute disparity maps using ssd template matching
    bs = 7; l = 0.0; dr = 100
    D_L_n = np.abs(disparity_ssd(L_n, R, block_size=bs, disparity_range=dr, lambda_factor=l))
    D_R_n = np.abs(disparity_ssd(R, L_n, block_size=bs, disparity_range=dr, lambda_factor=l))
    D_L_c = np.abs(disparity_ssd(L_c, R, block_size=bs, disparity_range=dr, lambda_factor=l))
    D_R_c = np.abs(disparity_ssd(R, L_c, block_size=bs, disparity_range=dr, lambda_factor=l))
    # shift and scale disparity maps
    L_n = cv2.normalize(L_n, L_n, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)
    D_L_n = cv2.normalize(D_L_n, D_L_n, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R_n = cv2.normalize(D_R_n, D_R_n, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_8U)
    L_c = cv2.normalize(L_c, L_c, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)
    D_L_c = cv2.normalize(D_L_c, D_L_c, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_8U)
    D_R_c = cv2.normalize(D_R_c, D_R_c, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_8U)
    #  revert dimensions by upsampling if previously downsampled
    if use_subsampling:
        L_n = cv2.pyrUp(L_n)
        L_c = cv2.pyrUp(L_c)
        D_L_n = cv2.pyrUp(D_L_n)
        D_L_c = cv2.pyrUp(D_L_c)
        D_R_n = cv2.pyrUp(D_R_n)
        D_R_c = cv2.pyrUp(D_R_c)
    # save disparity maps
    cv2.imwrite('output/pair1-L-noisy.png', L_n)
    cv2.imwrite('output/ps2-3-a-1.png', D_L_n)
    cv2.imwrite('output/ps2-3-a-2.png', D_R_n)
    cv2.imwrite('output/pair1-L-contrast.png', L_c)
    cv2.imwrite('output/ps2-3-b-1.png', D_L_c)
    cv2.imwrite('output/ps2-3-b-2.png', D_R_c)

# 4: Disparity map matching using normalized correlation
def ps2_4():
    # read real image pair in grayscale
    L = cv2.imread('input/pair1-L.png', cv2.IMREAD_GRAYSCALE)
    R = cv2.imread('input/pair1-R.png', cv2.IMREAD_GRAYSCALE)
    # downsample images and convert to [0,1] scale
    if use_subsampling:
        L = cv2.pyrDown(L)
        R = cv2.pyrDown(R)
    L = L * (1.0 / 255.0)
    R = R * (1.0 / 255.0)
    # distort left image using a) gaussian noise and b) increase in contrast
    noise = np.empty(L.shape, np.float)
    cv2.randn(noise, 0, 0.05);
    L_n = L + noise # add noise to left image
    L_c = L * 1.1  # increase contrast by 10% of left image
    # compute disparity maps
    bs = 7; l = 0.0; dr = 100
    D_L = np.abs(disparity_ncorr(L, R, block_size=bs, disparity_range=dr, lambda_factor=l))
    D_R = np.abs(disparity_ncorr(R, L, block_size=bs, disparity_range=dr, lambda_factor=l))
    D_L_n = np.abs(disparity_ncorr(L_n, R, block_size=bs, disparity_range=dr, lambda_factor=l))
    D_R_n = np.abs(disparity_ncorr(R, L_n, block_size=bs, disparity_range=dr, lambda_factor=l))
    D_L_c = np.abs(disparity_ncorr(L_c, R, block_size=bs, disparity_range=dr, lambda_factor=l))
    D_R_c = np.abs(disparity_ncorr(R, L_c, block_size=bs, disparity_range=dr, lambda_factor=l))
    # shift and scale disparity maps
    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)
    D_L_n = cv2.normalize(D_L_n, D_L_n, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R_n = cv2.normalize(D_R_n, D_R_n, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_L_c = cv2.normalize(D_L_c, D_L_c, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R_c = cv2.normalize(D_R_c, D_R_c, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #  revert dimensions by upsampling if previously downsampled
    if use_subsampling:
        D_L = cv2.pyrUp(D_L)
        D_R = cv2.pyrUp(D_R)
        D_L_n = cv2.pyrUp(D_L_n)
        D_R_n = cv2.pyrUp(D_R_n)
        D_L_c = cv2.pyrUp(D_L_c)
        D_R_c = cv2.pyrUp(D_R_c)
    # save disparity maps
    cv2.imwrite('output/ps2-4-a-1.png', D_L)
    cv2.imwrite('output/ps2-4-a-2.png', D_R)
    cv2.imwrite('output/ps2-4-b-1.png', D_L_n)
    cv2.imwrite('output/ps2-4-b-2.png', D_R_n)
    cv2.imwrite('output/ps2-4-b-3.png', D_L_c)
    cv2.imwrite('output/ps2-4-b-4.png', D_R_c)

# 5: Experiment with last image and smoothing/sharpening etc and comparing
#    the disparity maps with the ground truth
def ps2_5():
    # read real image pair in grayscale
    L = cv2.imread('input/pair2-L.png', cv2.IMREAD_GRAYSCALE)
    R = cv2.imread('input/pair2-R.png', cv2.IMREAD_GRAYSCALE)
    # equalize, smoothen, downsample and convert to [0,1] scale
    L = cv2.pyrDown(cv2.GaussianBlur(cv2.equalizeHist(L), (5,)*2, 3))
    R = cv2.pyrDown(cv2.GaussianBlur(cv2.equalizeHist(R), (5,)*2, 3))
    L = L * 1.0 / 255.0
    R = R * 1.0 / 255.0
    # compute disparity maps
    bs = 11; dr = 100; l = 0
    D_L = np.abs(disparity_ncorr(L, R, block_size=bs, disparity_range=dr, lambda_factor=l))
    D_R = np.abs(disparity_ncorr(R, L, block_size=bs, disparity_range=dr, lambda_factor=l))
    # shift and scale disparity maps
    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # upsample the disparity maps if previously downsampled
    if use_subsampling:
        D_L = cv2.pyrUp(D_L)
        D_R = cv2.pyrUp(D_R)
    #  sharpen the disparity maps
    #  kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    #  D_L = cv2.filter2D(D_L, -1, kernel)
    #  D_R = cv2.filter2D(D_R, -1, kernel)
    # equalize and smoothen disparity maps
    D_L = cv2.GaussianBlur(cv2.equalizeHist(D_L), (5,)*2, 3)
    D_R = cv2.GaussianBlur(cv2.equalizeHist(D_R), (5,)*2, 3)    # save disparity maps
    # write disparity maps to files
    cv2.imwrite('output/ps2-5-a-1.png', D_L)
    cv2.imwrite('output/ps2-5-a-2.png', D_R)

if __name__ == '__main__':
    ps2_list = [ps2_1, ps2_2, ps2_3, ps2_4, ps2_5]
    if len(sys.argv) == 2:
        if int(sys.argv[1]) in range(1,6):
            print('Executing task %d'%(int(sys.argv[1])))
            ps2_list[int(sys.argv[1])-1]()
        else:
            print('Give argument in range [1,5] for the corresponding tasks')
    else:
        print('Executing all tasks:')
        for idx, ps in enumerate(ps2_list):
            print('Executing task: %d'%(idx+1))
            ps()


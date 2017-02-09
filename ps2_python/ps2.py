# ps2
import os
import sys
import numpy as np
import cv2
from disparity_ssd import *
from disparity_ncorr import *

## 1: SSD match algorithm
def ps2_1():
    # Read images
    L = cv2.imread('input/pair0-L.png', cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)
    R = cv2.imread('input/pair0-R.png', cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)
    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    D_L = disparity_ssd(L, R, 15)
    D_R = disparity_ssd(R, L, 15)
    # shift and scale disparity maps
    D_L = D_L - D_L.min()
    D_L = D_L / D_L.max() * 255.0
    D_R = D_R - D_R.min()
    D_R = D_R / D_R.max() * 255.0
    # save disparity maps
    cv2.imwrite('output/ps2-1-a-1.png', D_L)
    cv2.imwrite('output/ps2-1-a-2.png', D_R)

# 2: Disparity map mathcing on a real image pair
def ps2_2():
    # read real image pair in grayscale
    L = cv2.imread('input/pair1-L.png', cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)
    R = cv2.imread('input/pair1-R.png', cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)
    # compute disparity maps
    D_L = disparity_ssd(L, R, 5)
    D_R = disparity_ssd(R, L, 5)
    # shift and scale disparity maps
    D_L = D_L - D_L.min()
    D_L = D_L / D_L.max() * 255.0
    D_R = D_R - D_R.min()
    D_R = D_R / D_R.max() * 255.0
    # save disparity maps
    cv2.imwrite('output/ps2-2-a-1.png', D_L)
    cv2.imwrite('output/ps2-2-a-2.png', D_R)
'''
The disparity maps produces using ssd differ a lot to the ground truth maps in
valua, while at the same time having smaller size due to the application of the
template only inside the boundaries
'''

# 3: Demonstration of the sensitivity of SSD to perturbations
def ps2_3():
    # read real image pair in grayscale
    L = cv2.imread('input/pair1-L.png', cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)
    R = cv2.imread('input/pair1-R.png', cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)
    # add noise to the left image
    noise = np.empty(L.shape, np.float)
    cv2.randn(noise, 0, 0.2);
    L_n = L + noise # add noise to left image
    L_c = L * 1.1  # increase contrast by 10% of left image
    cv2.normalize(L_n, L_n, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(L_c, L_c, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    # compute disparity maps
    D_L_n = disparity_ssd(L_n, R)
    D_R_n = disparity_ssd(R, L_n)
    D_L_c = disparity_ssd(L_c, R)
    D_R_c = disparity_ssd(R, L_c)
    # shift and scale disparity maps
    D_L_n = (D_L_n - D_L_n.min()) / D_L_n.max() * 255.0
    D_R_n = (D_R_n - D_R_n.min()) / D_R_n.max() * 255.0
    D_L_c = (D_L_c - D_L_c.min()) / D_L_c.max() * 255.0
    D_R_c = (D_R_c - D_R_c.min()) / D_R_c.max() * 255.0
    # save disparity maps
    cv2.imwrite('output/ps2-3-a-1.png', D_L_n)
    cv2.imwrite('output/ps2-3-a-2.png', D_R_n)
    cv2.imwrite('output/ps2-3-b-1.png', D_L_c)
    cv2.imwrite('output/ps2-3-b-2.png', D_R_c)

# 4: Disparity map matching using normalized correlation
def ps2_4():
    # read real image pair in grayscale
    L = cv2.imread('input/pair1-L.png', cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)
    R = cv2.imread('input/pair1-R.png', cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)
    # distort left image
    #  noise = np.empty(L.shape, np.float)
    #  cv2.randn(noise, 0, 0.2);
    #  L_n = L + noise # add noise to left image
    #  L_c = L * 1.1  # increase contrast by 10% of left image
    #  cv2.normalize(L_n, L_n, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    #  cv2.normalize(L_c, L_c, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    # compute disparity maps
    D_L = disparity_ncorr(L, R, template_size=15)
    #  D_R = disparity_ncorr(R, L)
    #  D_L_n = disparity_ncorr(L_n, R)
    #  D_R_n = disparity_ncorr(R, L_n)
    #  D_L_c = disparity_ncorr(L_c, R)
    #  D_R_c = disparity_ncorr(R, L_c)
    # shift and scale disparity maps
    D_L = (D_L - D_L.min()) / D_L.max() * 255.0
    #  D_R = (D_R - D_R.min()) / D_R.max() * 255.0
    #  D_L_n = (D_L_n - D_L_n.min()) / D_L_n.max() * 255.0
    #  D_R_n = (D_R_n - D_R_n.min()) / D_R_n.max() * 255.0
    #  D_L_c = (D_L_c - D_L_c.min()) / D_L_c.max() * 255.0
    #  D_R_c = (D_R_c - D_R_c.min()) / D_R_c.max() * 255.0
    # save disparity maps
    cv2.imwrite('output/ps2-4-a-1.png', D_L)
    #  cv2.imwrite('output/ps2-4-a-2.png', D_R)
    #  cv2.imwrite('output/ps2-4-a-1.png', D_L_n)
    #  cv2.imwrite('output/ps2-4-a-2.png', D_R_n)
    #  cv2.imwrite('output/ps2-4-b-1.png', D_L_c)
    #  cv2.imwrite('output/ps2-4-b-2.png', D_R_c)

# 5: Experiment with last image and smoothing/sharpening etc and comparing
#    the disparity maps with the ground truth
def ps2_5():
    pass

ps2_list = [ps2_1, ps2_2, ps2_3, ps2_4, ps2_5]
if len(sys.argv) == 2:
    print('Executing task %d'%(int(sys.argv[1])))
    ps2_list[int(sys.argv[1])-1]()
else:
    print('Executing all tasks:')
    for idx, ps in enumerate(ps2_list):
        print('Executing task: %d'%(idx+1))
        ps()


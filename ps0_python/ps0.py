#!/usr/bin/env python

import cv2
import numpy as np
import os.path

'''
Problem Set 0: Images as Functions
'''

if __name__ == '__main__':

    img1 = np.zeros((400,225,3), np.uint8)
    img2 = np.zeros((155,225,3), np.uint8)

    if not os.path.isfile('output/ps0-1-a-1.png')\
       or not os.path.isfile('output/ps0-1-a-2.png'):
        img1 = cv2.imread('input/lena.jpg', cv2.IMREAD_COLOR)
        cv2.imwrite('output/ps0-1-a-1.png', img1)
        img2 = cv2.imread('input/woman.tiff', cv2.IMREAD_COLOR)
        cv2.imwrite('output/ps0-1-a-2.png', img2)
    else:
        img1 = cv2.imread('output/ps0-1-a-1.png', cv2.IMREAD_COLOR)
        img2 = cv2.imread('output/ps0-1-a-2.png', cv2.IMREAD_COLOR)

    # Problem 2: Produce monochrome images using their channels
    # swap blue and red channels of wide image
    img1_swapped = img1.copy()
    img1_swapped[:,:,0], img1_swapped[:,:,2] =\
            img1_swapped[:,:,2], img1_swapped[:,:,0]
    # create monochrome images using the green and red channels of the orig. img
    img1_green = img1[:,:,1]
    img1_red = img1[:,:,2]
    img1_blue = img1[:,:,0]
    cv2.imwrite('output/ps0-2-a-1.png', img1_swapped)
    cv2.imwrite('output/ps0-2-b-1.png', img1_green)
    cv2.imwrite('output/ps0-2-c-1.png', img1_red)
    cv2.imwrite('output/ps0-2-d-1.png', img1_blue)


    # Problem 3: crop a 100x100 center region from mono img2 and paste in mono img1
    img1_center = [dim//2 for dim in img1.shape[:2]]
    img2_center = [dim//2 for dim in img2.shape[:2]]
    img1_mono = img1_green.copy();
    img2_mono = img2.copy(); img2_mono = img2_mono[:,:,1]
    img2_mono[img2_center[0]-50:img2_center[0]+50,
                  img2_center[1]-50:img2_center[1]+50] =\
            img1_mono[img1_center[0]-50:img1_center[0]+50,
                          img1_center[1]-50:img1_center[1]+50]
    img2_with_img1_crop = img2_mono
    cv2.imwrite('output/ps0-3-a-1.png', img2_with_img1_crop)

    # Problem 4:
    # a) Min and Max values of green image
    min1 = img1_green.min()
    max1 = img1_green.max()
    mean = img1_green.mean()
    std = img1_green.std()
    print('4a) img1_green: min=%d, max=%d, mean=%f, std=%f' % (min1, max1, mean,
                                                               std))
    # b) normalize img1_green and then multiply with 10 and add the mean
    img1_green_normed = img1_green.copy()
    img1_green_normed = cv2.add(cv2.multiply(cv2.divide(cv2.subtract(
        img1_green_normed, mean), std), 10), mean)
    cv2.imwrite('output/ps0-4-b-1.png', img1_green_normed)
    # c) shift img by 2 pixels to the left
    img1_shifted = img1_green_normed.copy()
    rows,cols = img1_shifted.shape
    M = np.float32([[1,0,-2],[0,1,0]])
    img1_shifted = cv2.warpAffine(img1_shifted, M, (cols, rows))
    cv2.imwrite('output/ps0-4-c-1.png', img1_shifted)
    # d) subtract shifted version from original
    img1_sub_shifted = img1_shifted - cv2.warpAffine(img1_shifted, M, (cols,
                                                                       rows))
    cv2.imwrite('output/ps0-4-d-1.png', img1_sub_shifted)

    # Problem 5: Gaussian Noise Addition
    # a) add Gaussian noise to green channel
    img_green_noise = img1.copy()
    height, width, depth = img_green_noise.shape
    noise = np.zeros((height,width), np.uint8)
    cv2.randn(noise, 0, 30)
    img_green_noise[:,:,1] = img_green_noise[:,:,1] + noise
    cv2.imwrite('output/ps0-5-a-1.png', img_green_noise)
    # b) add Gaussian noise to the blue channel
    img_blue_noise = img1.copy()
    img_blue_noise[:,:,0] = img_blue_noise[:,:,0] + noise
    cv2.imwrite('output/ps0-5-b-1.png', img_blue_noise)
    '''
    The same gaussian noise is more visible in the green channel than the blue,
    due to the increased sensitivity of the eye to the green color spectrum
    '''

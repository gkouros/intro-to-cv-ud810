import cv2
import numpy as np
import time
from hough_lines_acc import *
from hough_peaks import *
from hough_lines_draw import *

def ps1_3_fun():
    start_time = time.time()
    noisy_img = cv2.imread('input/ps1-input0-noise.png', cv2.IMREAD_GRAYSCALE)
    #  3a: smooth the noisy image using gaussian blurring
    smoothed_img = cv2.GaussianBlur(noisy_img, (23,)*2, 4.5)
    #  3b: perform edge detection on both images using Canny
    min_val = 20; max_val = 2 * min_val
    noisy_edge_img = cv2.Canny(noisy_img, min_val, max_val)
    edge_img = cv2.Canny(smoothed_img, min_val, max_val)
    #  3c: apply hough line detection to the smoothed image
    H, thetas, rhos = hough_lines_acc(edge_img)
    peaks = hough_peaks(H, numpeaks=20, threshold=50, nhood_size=150)
    for peak in peaks:
        cv2.circle(H, tuple(peak[::-1]), 5, (255,255,255), -1)
    color_img = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2BGR)
    hough_lines_draw(color_img, 'output/ps1-3-c-2.png', peaks, rhos, thetas)
    #  save the produced images
    cv2.imwrite('output/ps1-3-a-1.png', smoothed_img)
    cv2.imwrite('output/ps1-3-b-1.png', noisy_edge_img)
    cv2.imwrite('output/ps1-3-b-2.png', edge_img)
    cv2.imwrite('output/ps1-3-c-1.png', H)
    print('3) Time elapsed: %.2f s'%(time.time()-start_time))

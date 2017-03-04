import cv2
import numpy as np
import time
from hough_lines_acc import *
from hough_peaks import *
from hough_lines_draw import *
from auto_canny import *

def ps1_4():
    start_time = time.time()
    #  4a: Load the coin image, smooth it (gaussian blur) and save it
    img = cv2.imread('input/ps1-input1.png', cv2.IMREAD_GRAYSCALE)
    smoothed_img = cv2.GaussianBlur(img, (11,11), 3)
    cv2.imwrite('output/ps1-4-a-1.png', smoothed_img)
    #  4b: apply edge detection using Canny
    edge_img = auto_canny(smoothed_img, 0.8)
    cv2.imwrite('output/ps1-4-b-1.png', edge_img)
    #  4c: apply hough line detection to the smoothed image
    H, thetas, rhos = hough_lines_acc(edge_img)
    peaks = hough_peaks(H, numpeaks=10, threshold=150, nhood_size=20)
    for peak in peaks:
        cv2.circle(H, tuple(peak[::-1]), 15, (255,255,255), -1)
    cv2.imwrite('output/ps1-4-c-1.png', H)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hough_lines_draw(color_img, 'output/ps1-4-c-2.png', peaks, rhos, thetas)
    print('4) Time elapsed: %.2f s'%(time.time()-start_time))

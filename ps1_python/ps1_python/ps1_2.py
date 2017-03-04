import cv2
import numpy as np
import time
from hough_lines_acc import *
from hough_peaks import *
from hough_lines_draw import *
from auto_canny import *

def ps1_2():
    start_time = time.time()
    # load input image and perform edge detection
    img = cv2.imread('input/ps1-input0.png', cv2.IMREAD_GRAYSCALE)
    edge_img = auto_canny(img, 0.5)
    # 2-a: Calculate the hough space of the edge image for line detection
    H, thetas, rhos = hough_lines_acc(edge_img)
    cv2.imwrite('output/ps1-2-a-1.png', H)
    # 2-b: Detect peaks on the hough space of the edge image and highlight them
    peaks = hough_peaks(H, numpeaks=10, threshold=100, nhood_size=50)
    H_peaks = H.copy()
    for peak in peaks:
        cv2.circle(H_peaks, tuple(peak[::-1]), 5, (255,255,255), -1)
    # 2-c: Draw the detected lines on the image
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hough_lines_draw(color_img, 'output/ps1-2-c-1.png', peaks, rhos, thetas)
    print('2) Time elapsed: %.2f s'%(time.time()-start_time))

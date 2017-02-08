import cv2
import numpy as np
import time
from auto_canny import *
from hough_lines_acc import *
from hough_peaks import *
from hough_lines_draw import *
from filter_lines import *

def ps1_6_fun():
    start_time = time.time()
    img = cv2.imread('input/ps1-input2.png', cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smoothed_img = cv2.GaussianBlur(gray_img, (7,7), 5)
    edge_img = cv2.Canny(smoothed_img, 50, 100)
    H, thetas, rhos = hough_lines_acc(edge_img)
    peaks = hough_peaks(H, numpeaks=10, threshold=120, nhood_size=50)
    hl_img = cv2.cvtColor(smoothed_img, cv2.COLOR_GRAY2BGR)
    hough_lines_draw(hl_img, 'output/ps1-6-a-1.png', peaks, rhos, thetas)
    peaks = filter_lines(peaks, thetas, rhos, 5, 50)
    hl_img2 = cv2.cvtColor(smoothed_img, cv2.COLOR_GRAY2BGR)
    hough_lines_draw(hl_img2, 'output/ps1-6-c-1.png', peaks, rhos, thetas)
    print('6) Time elapsed: %.2f s'%(time.time()-start_time))

import cv2
import numpy as np
import time
from auto_canny import *
from hough_lines_acc import *
from hough_peaks import *
from hough_lines_draw import *
from filter_lines import *
from find_circles import *
from hough_circles_draw import *

def ps1_8_fun():
    start_time = time.time()
    img = cv2.imread('input/ps1-input3.png', cv2.IMREAD_COLOR)
    smoothed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smoothed_img = cv2.erode(smoothed_img, np.ones((3,)*2,np.uint8), 1)
    smoothed_img = cv2.GaussianBlur(smoothed_img, (3,)*2, 2)
    edge_img = cv2.Canny(smoothed_img, 40, 80)

    #  Detect lines
    H, thetas, rhos = hough_lines_acc(edge_img)
    peaks = hough_peaks(H, numpeaks=40, threshold=105, nhood_size=40)
    peaks = filter_lines(peaks, thetas, rhos, 3, 24)
    img_hl = hough_lines_draw(img, 'output/ps1-8-a-1.png', peaks, rhos, thetas)

    #  Detect circles
    centers, radii = find_circles(edge_img, [20, 40], threshold=110, nhood_size=50)
    img_circles = img.copy()
    for i in range(len(radii)):
        img_hl = hough_circles_draw(img_hl, 'output/ps1-8-a-1.png',
                                         centers[i], radii[i])
    print('\033[F\r8) Time elapsed: %.2f s'%(time.time()-start_time))

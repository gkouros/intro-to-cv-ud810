import cv2
import numpy as np
import time
from hough_peaks import *
from find_circles import *
from hough_circles_draw import *
from auto_canny import *

def ps1_5_fun():
    start_time = time.time()
    # 5a: Load coin image, smooth, detect edges and calculate hough space
    img = cv2.imread('input/ps1-input1.png', cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smoothed_img = cv2.GaussianBlur(gray_img, (11,11), 3)  # smooth image
    edge_img = auto_canny(smoothed_img, 0.8)
    cv2.imwrite('output/ps1-5-a-1.png', smoothed_img)
    cv2.imwrite('output/ps1-5-a-2.png', edge_img)
    # detect circles with radius = 20 and save image
    H_20 = hough_circles_acc(edge_img, 20)
    peaks = hough_peaks(H_20, numpeaks=10, threshold=140, nhood_size=100)
    img_circles = img.copy()
    img_circles = hough_circles_draw(img_circles, 'output/ps1-5-a-3.png', peaks, 20)
    # 5b: detect circles in the range [20 50]
    centers, radii = find_circles(edge_img, [20, 50], threshold=153, nhood_size=10)
    img_circles = img.copy()
    for i in range(len(radii)):
        img_circles = hough_circles_draw(img_circles, 'output/ps1-5-b-1.png',
                                         centers[i], radii[i])
    print('\033[F\r5) Time elapsed: %.2f s'%(time.time()-start_time))

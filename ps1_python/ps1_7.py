import cv2
import numpy as np
import time
from hough_peaks import *
from find_circles import *
from hough_circles_draw import *
from auto_canny import *

def ps1_7_fun():
    start_time = time.time()
    img = cv2.imread('input/ps1-input2.png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eroded_img = cv2.erode(gray_img, np.ones((5,5),np.uint8), 1)
    smoothed_img = cv2.blur(eroded_img, (3,3))
    edge_img = auto_canny(smoothed_img, 0.5)
    centers, radii = find_circles(edge_img, [20, 40], threshold=135, nhood_size=10)
    img_circles = img.copy()
    for i in range(len(radii)):
        img_circles = hough_circles_draw(img_circles, 'output/ps1-7-a-1.png',
                                         centers[i], radii[i])
    print('\033[F\r7) Time elapsed: %.2f'%(time.time()-start_time))

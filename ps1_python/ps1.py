#!/usr/bin/env python3

import cv2
import numpy as np
from hough_lines_acc import *
from hough_peaks import *

# 1-a
img = cv2.imread('input/ps1-input0.png', cv2.IMREAD_GRAYSCALE)
# compute edge image img_edges using canny
#  img = cv2.blur(img, (3,3)) # blur image before edge detection
min_val = 100; max_val = 2 * min_val
img_edges = cv2.Canny(img, min_val, max_val)
cv2.imwrite('output/ps1-1-a-1.png', img_edges)
#  cv2.imshow('checkerboard', img)
#  cv2.waitKey(0)
#  cv2.imshow('edges of checkerboard', img_edges)

# 2-a: Calculate the hough space of the edge image
# plot/show accumulator array H and save as output/ps1-2-a-1.png
H, theta, rho = hough_lines_acc(img_edges)
cv2.imwrite('output/ps1-2-a-1.png', H)
cv2.imshow('Accumulator array H', H)
cv2.waitKey(0)

# 2-b: Detect peaks on the hough space of the edge image
peaks = hough_peaks(H, 10)
# highlight peak locations on accumulator array
H_peaks = H.copy()
for peak in peaks:
    cv2.circle(H_peaks, tuple(peak), 5, (255,255,255), -1)
# save as output/ps1-2-b-1.png
cv2.imwrite('output/ps1-2-b-1.png', H_peaks)
cv2.imshow('Accumulator array H with highlighted peaks', H_peaks)

# 3: Perform Hough line detection on a noisy input image


cv2.waitKey(0)

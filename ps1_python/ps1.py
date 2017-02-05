#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
from hough_lines_acc import *
from hough_peaks import *
from hough_lines_draw import *

fig = plt.figure()

# 1-a
img = cv2.imread('input/ps1-input0.png', cv2.IMREAD_GRAYSCALE)
# compute edge image img_edges using canny
#  img = cv2.blur(img, (3,3)) # blur image before edge detection
min_val = 100; max_val = 2 * min_val
img_edges = cv2.Canny(img, min_val, max_val)
cv2.imwrite('output/ps1-1-a-1.png', img_edges)


p1a = fig.add_subplot(3, 3, 1)
p1a.imshow(img, cmap='gray')
p1a.set_title('Input Image')
p1a2 = fig.add_subplot(3,3,2)
p1a2.imshow(img_edges, cmap='gray')
p1a2.set_title('Edge Image')

# 2-a: Calculate the hough space of the edge image
# plot/show accumulator array H and save as output/ps1-2-a-1.png
H, thetas, rhos = hough_lines_acc(img_edges)
cv2.imwrite('output/ps1-2-a-1.png', H)
p2a = fig.add_subplot(3,3,3)
p2a.imshow(H, cmap='gray')
p2a.set_title('Hough Space for Lines')

# 2-b: Detect peaks on the hough space of the edge image and highlight them
peaks = hough_peaks(H, numpeaks=10, threshold=100, nhood_size=1)
H_peaks = H.copy()
for peak in peaks:
    cv2.circle(H_peaks, tuple(peak[::-1]), 5, (255,255,255), -1)
cv2.imwrite('output/ps1-2-b-1.png', H_peaks)  # save as output/ps1-2-b-1.png
p2b = fig.add_subplot(3,3,4)
p2b.imshow(H_peaks, cmap='gray')
p2b.set_title('Hough Space with peaks')

# 2-c: Draw the detected lines on the image
color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
hough_lines_draw(color_img, 'output/ps1-2-c-1.png', peaks, rhos, thetas)
cv2.waitKey(0)
p2c = fig.add_subplot(3,3,5)
p2c.imshow(color_img)
p2c.set_title('Input image and lines')

plt.show()

# 3: Perform Hough line detection on a noisy input image



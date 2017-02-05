#!/usr/bin/env python3

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from hough_lines_acc import *
from hough_peaks import *
from hough_lines_draw import *

fig = plt.figure()

# 1-a
#  img = cv2.imread('input/ps1-input0.png', cv2.IMREAD_GRAYSCALE)
# compute edge image img_edges using canny
#  min_val = 100; max_val = 2 * min_val
#  img_edges = cv2.Canny(img, min_val, max_val)
#  cv2.imwrite('output/ps1-1-a-1.png', img_edges)

#  p1a = fig.add_subplot(3, 3, 1)
#  p1a.imshow(img, cmap='gray')
#  p1a.set_title('Input Image')
#  p1a2 = fig.add_subplot(4,4,2)
#  p1a2.imshow(img_edges, cmap='gray')
#  p1a2.set_title('Edge Image')

# 2-a: Calculate the hough space of the edge image
# plot/show accumulator array H and save as output/ps1-2-a-1.png
#  H, thetas, rhos = hough_lines_acc(img_edges)
#  cv2.imwrite('output/ps1-2-a-1.png', H)
# 2-b: Detect peaks on the hough space of the edge image and highlight them
#  peaks = hough_peaks(H, numpeaks=10, threshold=100, nhood_size=50)
#  H_peaks = H.copy()
#  for peak in peaks:
    #  cv2.circle(H_peaks, tuple(peak[::-1]), 5, (255,255,255), -1)
# 2-c: Draw the detected lines on the image
#  color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#  hough_lines_draw(color_img, 'output/ps1-2-c-1.png', peaks, rhos, thetas)

#  p2a = fig.add_subplot(4,4,3)
#  p2a.imshow(H, cmap='gray')
#  p2a.set_title('Hough Space for Lines')
#  cv2.imwrite('output/ps1-2-b-1.png', H_peaks)  # save as output/ps1-2-b-1.png
#  p2b = fig.add_subplot(4,4,4)
#  p2b.imshow(H_peaks, cmap='gray')
#  p2b.set_title('Hough Space with peaks')
#  p2c = fig.add_subplot(4,4,5)
#  p2c.imshow(color_img)
#  p2c.set_title('Input image and lines')


# 3: Perform Hough line detection on a noisy input image
#  noisy_img = cv2.imread('input/ps1-input0-noise.png', cv2.IMREAD_GRAYSCALE)
# 3a: smooth the noisy image using gaussian blurring
#  smoothed_img = cv2.GaussianBlur(noisy_img, (21,21), 5)
# 3b: perform edge detection on both images using Canny
#  min_val = 20; max_val = 2 * min_val
#  noisy_img_edges = cv2.Canny(noisy_img, min_val, max_val)
#  smoothed_img_edges = cv2.Canny(smoothed_img, min_val, max_val)
#  3c: apply hough line detection to the smoothed image
#  H, thetas, rhos = hough_lines_acc(smoothed_img_edges)
#  peaks = hough_peaks(H, numpeaks=20, threshold=50, nhood_size=150)
#  for peak in peaks:
    #  cv2.circle(H, tuple(peak[::-1]), 5, (255,255,255), -1)
#  color_img = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2RGB)
#  hough_lines_draw(color_img, 'output/ps1-3-c-2.png', peaks, rhos, thetas)

#  save the produced images
#  cv2.imwrite('output/ps1-3-a-1.png', smoothed_img)
#  cv2.imwrite('output/ps1-3-b-1.png', noisy_img_edges)
#  cv2.imwrite('output/ps1-3-b-2.png', smoothed_img_edges)
#  cv2.imwrite('output/ps1-3-c-1.png', H)

#  plot the images on their respective subplots
#  p3a0 = fig.add_subplot(4,4,6)
#  p3a0.imshow(noisy_img, cmap='gray')
#  p3a0.set_title('Noisy input image')
#  p3a1 = fig.add_subplot(4,4,7)
#  p3a1.imshow(smoothed_img, cmap='gray')
#  p3a1.set_title('Smoothed input image')
#  p3b1 = fig.add_subplot(4,4,8)
#  p3b1.imshow(noisy_img_edges, cmap='gray')
#  p3b1.set_title('Noisy image edges')
#  p3b2 = fig.add_subplot(4,4,9)
#  p3b2.imshow(smoothed_img_edges, cmap='gray')
#  p3b2.set_title('Smoothed image edges')
#  p3c1 = fig.add_subplot(4,4,10)
#  p3c1.imshow(H, cmap='gray')
#  p3c1.set_title('Smoothed image Hough space')
#  p3c2 = fig.add_subplot(4,4,11)
#  p3c2.imshow(color_img)
#  p3c2.set_title('Noisy image lines')

# 4: Similarly to 3 but using the coin image
# 4a: Load the coin image, smooth it (gaussian blur) and save it
img = cv2.imread('input/ps1-input1.png', cv2.IMREAD_GRAYSCALE)
smoothed_img = cv2.GaussianBlur(img, (11,11), 5)
cv2.imwrite('output/ps1-4-a-1.png', smoothed_img)
# 4b: apply edge detection using Canny
smoothed_img_edges = cv2.Canny(img, 100, 900)
cv2.imwrite('output/ps1-4-b-1.png', smoothed_img_edges)
# 3c: apply hough line detection to the smoothed image
H, thetas, rhos = hough_lines_acc(smoothed_img_edges)
peaks = hough_peaks(H, numpeaks=10, threshold=150, nhood_size=10)
for peak in peaks:
    cv2.circle(H, tuple(peak[::-1]), 5, (255,255,255), -1)
cv2.imwrite('output/ps1-4-c-1.png', H)
color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
hough_lines_draw(color_img, 'output/ps1-4-c-2.png', peaks, rhos, thetas)

#  plt.subplots_adjust(hspace=0.75)
#  plt.show()

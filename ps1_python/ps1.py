#!/usr/bin/env python3

import sys
import time
import cv2
import numpy as np
from hough_lines_acc import *
from hough_peaks import *
from hough_lines_draw import *
from find_circles import *
from hough_circles_draw import *
from filter_lines import *
from auto_canny import *

# 1-a: compute edge image edge_img using canny
#  img = cv2.imread('input/ps1-input0.png', cv2.IMREAD_GRAYSCALE)
#  min_val = 100; max_val = 2 * min_val
#  edge_img = cv2.Canny(img, min_val, max_val)
#  cv2.imwrite('output/ps1-1-a-1.png', edge_img)

# 2-a: Calculate the hough space of the edge image
#  plot/show accumulator array H and save as output/ps1-2-a-1.png
#  H, thetas, rhos = hough_lines_acc(edge_img)
#  cv2.imwrite('output/ps1-2-a-1.png', H)
# 2-b: Detect peaks on the hough space of the edge image and highlight them
#  peaks = hough_peaks(H, numpeaks=10, threshold=100, nhood_size=50)
#  H_peaks = H.copy()
#  for peak in peaks:
    #  cv2.circle(H_peaks, tuple(peak[::-1]), 5, (255,255,255), -1)
# 2-c: Draw the detected lines on the image
#  color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#  hough_lines_draw(color_img, 'output/ps1-2-c-1.png', peaks, rhos, thetas)

# 3: Perform Hough line detection on a noisy input image
#  noisy_img = cv2.imread('input/ps1-input0-noise.png', cv2.IMREAD_GRAYSCALE)
# 3a: smooth the noisy image using gaussian blurring
#  smoothed_img = cv2.GaussianBlur(noisy_img, (21,21), 5)
# 3b: perform edge detection on both images using Canny
#  min_val = 20; max_val = 2 * min_val
#  noisy_edge_img = cv2.Canny(noisy_img, min_val, max_val)
#  edge_img = cv2.Canny(smoothed_img, min_val, max_val)
# 3c: apply hough line detection to the smoothed image
#  H, thetas, rhos = hough_lines_acc(edge_img)
#  peaks = hough_peaks(H, numpeaks=20, threshold=50, nhood_size=150)
#  for peak in peaks:
    #  cv2.circle(H, tuple(peak[::-1]), 5, (255,255,255), -1)
#  color_img = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2BGR)
#  hough_lines_draw(color_img, 'output/ps1-3-c-2.png', peaks, rhos, thetas)
#  save the produced images
#  cv2.imwrite('output/ps1-3-a-1.png', smoothed_img)
#  cv2.imwrite('output/ps1-3-b-1.png', noisy_edge_img)
#  cv2.imwrite('output/ps1-3-b-2.png', edge_img)
#  cv2.imwrite('output/ps1-3-c-1.png', H)

# 4: Similarly to 3 but using the coin image
# 4a: Load the coin image, smooth it (gaussian blur) and save it
#  img = cv2.imread('input/ps1-input1.png', cv2.IMREAD_GRAYSCALE)
#  smoothed_img = cv2.GaussianBlur(img, (11,11), 5)
#  cv2.imwrite('output/ps1-4-a-1.png', smoothed_img)
#  4b: apply edge detection using Canny
#  edge_img = cv2.Canny(img, 100, 900)
#  cv2.imwrite('output/ps1-4-b-1.png', edge_img)
#  3c: apply hough line detection to the smoothed image
#  H, thetas, rhos = hough_lines_acc(edge_img)
#  peaks = hough_peaks(H, numpeaks=10, threshold=150, nhood_size=10)
#  for peak in peaks:
    #  cv2.circle(H, tuple(peak[::-1]), 5, (255,255,255), -1)
#  cv2.imwrite('output/ps1-4-c-1.png', H)
#  color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#  hough_lines_draw(color_img, 'output/ps1-4-c-2.png', peaks, rhos, thetas)

# 5: Hough Transform for Circle Detection
# 5a: Load coin image, smooth, detect edges and calculate hough space
#  img = cv2.imread('input/ps1-input1.png', cv2.IMREAD_COLOR)
#  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#  smoothed_img = cv2.GaussianBlur(gray_img, (9,9), 3)  # smooth image
#  cv2.imwrite('output/ps1-5-a-1.png', smoothed_img)
#  edge_img = auto_canny(smoothed_img, 0.5)
#  cv2.imwrite('output/ps1-5-a-2.png', edge_img)

# detect circles with radius = 20 and save image
#  H_20 = hough_circles_acc(edge_img, 20)
#  peaks = hough_peaks(H_20, numpeaks=10, threshold=140, nhood_size=100)
#  img_circles = img.copy()
#  img_circles = hough_circles_draw(img_circles, 'output/ps1-5-a-3.png', peaks, 20)

# 5b: detect circles in the range [20 50]
#  start_time = time.time()
#  centers, radii = find_circles(edge_img, [20, 50], threshold=153, nhood_size=10)
#  img_circles = img.copy()
#  for i in range(len(radii)):
    #  img_circles = hough_circles_draw(img_circles, 'output/ps1-5-b-1.png',
                                     #  centers[i], radii[i])
#  print('Time elapsed: %f', time.time()-start_time)

# 6: Apply line detection on image with cluttering
#  img = cv2.imread('input/ps1-input2.png', cv2.IMREAD_COLOR)
#  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#  smoothed_img = cv2.GaussianBlur(gray_img, (7,7), 5)
#  edge_img = cv2.Canny(smoothed_img, 50, 100)
#  H, thetas, rhos = hough_lines_acc(edge_img)
#  peaks = hough_peaks(H, numpeaks=10, threshold=120, nhood_size=50)
#  hl_img = cv2.cvtColor(smoothed_img, cv2.COLOR_GRAY2BGR)
#  hough_lines_draw(hl_img, 'output/ps1-6-a-1.png', peaks, rhos, thetas)
#  peaks = filter_lines(peaks, thetas, rhos, 5, 50)
#  hl_img2 = cv2.cvtColor(smoothed_img, cv2.COLOR_GRAY2BGR)
#  hough_lines_draw(hl_img2, 'output/ps1-6-c-1.png', peaks, rhos, thetas)

# 7: Apply Hough circle detection on the cluttered image
#  img = cv2.imread('input/ps1-input2.png')
#  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#  eroded_img = cv2.erode(gray_img, np.ones((5,5),np.uint8), 1)
#  smoothed_img = cv2.blur(eroded_img, (3,3))
#  edge_img = auto_canny(smoothed_img, 0.5)
#  start_time = time.time()
#  centers, radii = find_circles(edge_img, [20, 40], threshold=135, nhood_size=10)
#  img_circles = img.copy()
#  for i in range(len(radii)):
    #  img_circles = hough_circles_draw(img_circles, 'output/ps1-7-a-1.png',
                                     #  centers[i], radii[i])
#  print('Time elapsed: %f', time.time()-start_time)

# 8: Apply line and circle detection on the distorted cluttered image
img = cv2.imread('input/ps1-input3.png', cv2.IMREAD_COLOR)
smoothed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
smoothed_img = cv2.erode(smoothed_img, np.ones((3,)*2,np.uint8), 1)
smoothed_img = cv2.GaussianBlur(smoothed_img, (3,)*2, 2)
edge_img = cv2.Canny(smoothed_img, 40, 80)
#  cv2.imshow("smooth and edge images", np.hstack([smoothed_img, edge_img]))
#  cv2.waitKey(0); cv2.destroyAllWindows()

start_time = time.time()
# Detect lines
H, thetas, rhos = hough_lines_acc(edge_img)
peaks = hough_peaks(H, numpeaks=40, threshold=105, nhood_size=40)
peaks = filter_lines(peaks, thetas, rhos, 3, 24)
img_hl = hough_lines_draw(img, 'output/ps1-8-a-1.png', peaks, rhos, thetas)
#  cv2.imwrite('output/temp.png', np.hstack([edge_img, img_lines[:,:,1]]))
#  cv2.imshow("smooth and edge images", np.hstack([smoothed_img, edge_img, img_lines[:,:,1]]))
#  cv2.waitKey(0); cv2.destroyAllWindows()

# Detect circles
centers, radii = find_circles(edge_img, [20, 40], threshold=110, nhood_size=50)
img_circles = img.copy()
for i in range(len(radii)):
    img_hl = hough_circles_draw(img_hl, 'output/ps1-8-a-1.png',
                                     centers[i], radii[i])
#  cv2.imwrite('output/temp.png', np.hstack([edge_img, img_circles[:,:,1]]))
print('Time elapsed: %f', time.time()-start_time)

#  TODO: try to fix the circle problem (circle -> ellipsis) (line -> line -> no problem)

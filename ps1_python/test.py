import cv2
import numpy as np
from time import sleep

def nothing(x):
    pass

img = cv2.imread('input/ps1-input2.png', cv2.IMREAD_GRAYSCALE)
cv2.namedWindow('image')
cv2.createTrackbar('d', 'image', 0, 20, nothing)
cv2.createTrackbar('sigma_color', 'image', 0, 1000, nothing)
cv2.createTrackbar('sigma_space', 'image', 0, 1000, nothing)

while 1:
    d = cv2.getTrackbarPos('d','image')
    sigma_color = cv2.getTrackbarPos('sigma_color','image')
    sigma_space = cv2.getTrackbarPos('sigma_space','image')
    smoothed_img = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    cv2.imshow('image', smoothed_img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    sleep(0.1)

cv2.destroyAllWindows()

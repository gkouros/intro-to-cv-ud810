import cv2
import numpy as np

def hough_circles_draw(img, outfile, peaks, radius):
    for peak in peaks:
        cv2.circle(img, tuple(peak[::-1]), radius, (0,255,0), 2)
    cv2.imwrite(outfile, img)
    return img

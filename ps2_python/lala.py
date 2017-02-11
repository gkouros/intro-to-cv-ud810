import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('input/pair1-L.png',0)
imgR = cv2.imread('input/pair1-R.png',0)

stereo = cv2.StereoSGBM_create(minDisparity=32, numDisparities=80, blockSize=5,
                               uniquenessRatio = 10,
                               speckleWindowSize = 100,
                               speckleRange = 32,
                               disp12MaxDiff = 1,
                               P1 = 8*3*5**2,
                               P2 = 32*3*5**2,
                               )
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()

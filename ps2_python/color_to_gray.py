import cv2
import numpy as np

L1 = cv2.imread('input/pair1-L.png', cv2.IMREAD_GRAYSCALE)
R1 = cv2.imread('input/pair1-R.png', cv2.IMREAD_GRAYSCALE)
L2 = cv2.imread('input/pair2-L.png', cv2.IMREAD_GRAYSCALE)
R2 = cv2.imread('input/pair2-R.png', cv2.IMREAD_GRAYSCALE)

cv2.imwrite('output/pair1-L-gray.png', L1)
cv2.imwrite('output/pair1-R-gray.png', R1)
cv2.imwrite('output/pair2-L-gray.png', L2)
cv2.imwrite('output/pair2-R-gray.png', R2)

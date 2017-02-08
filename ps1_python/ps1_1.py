import cv2
import numpy as np
import time
from auto_canny import *

def ps1_1_fun():
    start_time = time.time()
    img = cv2.imread('input/ps1-input0.png', cv2.IMREAD_GRAYSCALE)
    edge_img = auto_canny(img, 0.5)
    cv2.imwrite('output/ps1-1-a-1.png', edge_img)
    print('1) Time elapsed: %.2f s'%(time.time()-start_time))

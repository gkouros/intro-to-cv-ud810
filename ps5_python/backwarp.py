import cv2
import numpy as np

def backwarp(img, flow):
    h, w = flow.shape[:2]
    flow_map = -flow.copy()
    flow_map[:,:,0] += np.arange(w)
    flow_map[:,:,1] += np.arange(h)[:,np.newaxis]
    warped = cv2.remap(img, flow_map, None, cv2.INTER_LINEAR)
    return warped

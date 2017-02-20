import cv2
import numpy as np

def convert_flow_to_map(flow_x, flow_y):
    map_y, map_x = np.mgrid[:flow_x.shape[0], :flow_x.shape[1]]
    map_x = map_x.astype('float32')
    map_y = map_y.astype('float32')
    map_x += flow_x
    map_y += flow_y
    return map_x, map_y

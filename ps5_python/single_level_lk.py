import cv2
import numpy as np
from lk_optic_flow import *
from vis_optic_flow import *
from gl_pyramids import *
from backwarp import *

def single_level_lk(frs, levels, window):
    gpyrs = [gaussian_pyramid(frs[i], levels) for i in range(len(frs))]
    #  flows = [cv2.calcOpticalFlowFarneback(gpyrs[i][0], gpyrs[i+1][0], None,
                                           #  0.5, 3, window, 3, 5, 1.2, 0)
             #  for i in range(len(frs)-1)]
    flows = [-lk_optic_flow(frs[i], frs[i+1], window)#gpyrs[i][-1], gpyrs[i+1][-1], window)
             for i in range(len(frs)-1)]
    #  for i in range(len(flows)):
        #  for j in range(levels-1):
            #  flows[i] = expand(flows[i]) * 2
        #  resize flow matrix to the dimensions of the frame
        #  flows[i] = flows[i][:frs[i].shape[0], :frs[i].shape[1]]
        #  if flows[i].shape[:2] != frs[i].shape[:2]:
            #  flows[i] = cv2.resize(flows[i], (frs[i].shape[::-1]))
        #  flows[i][...,:]=1
    warps = [backwarp(fr, flow) for fr, flow in zip(frs[1:], flows)]
    diffs = [cv2.subtract(frs[i], warps[i]) for i in range(len(warps))]
    diffs = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in diffs]
    return flows, warps, diffs

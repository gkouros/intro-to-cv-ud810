import numpy as np
import random
from least_squares_M_solver import *
from svd_M_solver import *

def calc_residual(pts_2d, pts_3d, M):
    pts_2d_proj = np.array([np.dot(M, np.append(pt_3d,1)) for pt_3d in pts_3d])
    pts_2d_proj = pts_2d_proj[:,:2] / pts_2d_proj[:,2:]#.reshape(4,1)
    res = np.linalg.norm(pts_2d - pts_2d_proj)
    return res

def best_M(pts_2d, pts_3d, num_calibration_pts, num_test_pts, iterations):
    num_pts = pts_2d.shape[0]
    M = np.zeros((3,4), dtype=np.float32)
    res = 1e9
    for iter in range(iterations):
        idxs = random.sample(range(num_pts), num_calibration_pts)
        M_tmp,_ = least_squares_M_solver(pts_2d[idxs], pts_3d[idxs])
        #  M_tmp = svd_M_solver(pts_2d[idxs], pts_3d[idxs])
        test_idxs = [i for i in range(num_pts) if i not in idxs]
        test_idxs = random.sample(test_idxs, num_test_pts)
        res_tmp = calc_residual(pts_2d[test_idxs], pts_3d[test_idxs], M_tmp)
        if res_tmp < res:
            res = res_tmp
            M = M_tmp
    return M, res

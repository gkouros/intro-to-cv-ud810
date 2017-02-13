import numpy as np

def least_squares_F_solver(pts_a, pts_b):
    num_pts = pts_a.shape[0]
    ua = pts_a[:,0]
    va = pts_a[:,1]
    ub = pts_b[:,0]
    vb = pts_b[:,1]
    ones = np.ones(num_pts)
    A = np.column_stack((ua*ub, va*ub, ub, ua*vb, va*vb, vb, ua, va))
    b = -np.ones(num_pts)
    F = np.linalg.lstsq(A, b)[0]
    F = np.append(F, 1)
    F = F.reshape((3,3))
    return F

import numpy as np

def svd_F_solver(pts_a, pts_b):
    num_pts = pts_a.shape[0]
    ua = pts_a[:,0]
    va = pts_a[:,1]
    ub = pts_b[:,0]
    vb = pts_b[:,1]
    ones = np.ones(num_pts)
    A = np.column_stack((ua*ub, va*ub, ub, ua*vb, va*vb, vb, ua, va, ones))
    _,_,V = np.linalg.svd(A, full_matrices=True)
    F = V.T[:,-1]
    F = F.reshape((3,3))
    return F

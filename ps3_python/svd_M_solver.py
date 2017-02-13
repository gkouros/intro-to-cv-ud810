import numpy as np

def svd_M_solver(pts_2d, pts_3d):
    #  M = np.zeros((12,1), dtype=np.float32)
    num_pts = pts_2d.shape[0]
    A = np.zeros((2*num_pts,12), dtype=np.float32)
    b = np.zeros(2*num_pts, dtype=np.float32)
    x = pts_2d[:,0]
    y = pts_2d[:,1]
    X = pts_3d[:,0]
    Y = pts_3d[:,1]
    Z = pts_3d[:,2]
    zeros = np.zeros(num_pts)
    ones = np.ones(num_pts)
    A[::2,:]   = np.column_stack((X, Y, Z, ones, zeros, zeros, zeros, zeros, -x*X, -x*Y, -x*Z, -x))
    A[1::2,:] = np.column_stack((zeros, zeros, zeros, zeros, X, Y, Z, ones, -y*X, -y*Y, -y*Z, -y))
    _,_,V = np.linalg.svd(A, full_matrices=True)
    M = V.T[:,-1]
    M = M.reshape((3,4))
    return M

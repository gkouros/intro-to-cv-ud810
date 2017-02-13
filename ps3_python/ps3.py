# ps3
import cv2
import numpy as np
import sys
import random
from collections import OrderedDict
from load_file import *
from least_squares_M_solver import *
from svd_M_solver import *
from best_M import *
from least_squares_F_solver import *
from svd_F_solver import *

M_norm_a = np.array([[-0.4583, 0.2947, 0.0139, -0.0040],
                     [0.0509, 0.0546, 0.5410, 0.0524],
                     [-0.1090, -0.1784, 0.0443, -0.5968]], dtype=np.float32)

def ps3_1_a():
    # a) estimate camera projection matrix
    pts_2d = load_file('input/pts2d-norm-pic_a.txt')
    pts_3d = load_file('input/pts3d-norm.txt')
    # test results using a least squares solver
    M, res = least_squares_M_solver(pts_2d, pts_3d)
    pt_2d_proj = np.dot(M, np.append(pts_3d[-1],1))
    pt_2d_proj = pt_2d_proj[:2] / pt_2d_proj[2]
    res = np.linalg.norm(pts_2d[-1] - pt_2d_proj)
    print('Results with least squares:')
    print('M=%s'%M)
    print('Point %s projected to point %s'%(pts_2d[-1] ,pt_2d_proj))
    print('Residual: %.4f\n'%res)
    # test results using a least squares solver
    M = svd_M_solver(pts_2d, pts_3d)
    pt_2d_proj = np.dot(M, np.append(pts_3d[-1],1))
    pt_2d_proj = pt_2d_proj[:2] / pt_2d_proj[2]
    res = np.linalg.norm(pts_2d[-1] - pt_2d_proj)
    print('Results with SVD:')
    print('M=%s'%M)
    print('Point %s projected to point %s'%(pts_2d[-1] ,pt_2d_proj))
    print('Residual: %.4f\n'%res)


def ps3_1_b():
    # a) estimate camera projection matrix
    pts_2d = load_file('input/pts2d-pic_a.txt')
    pts_3d = load_file('input/pts3d.txt')
    # test results using a least squares solver for dif. number of calibration
    # points for 10 iterations
    M_8, res_8 = best_M(pts_2d, pts_3d, num_calibration_pts=8,
                        num_test_pts=4, iterations=10)
    M_12, res_12 = best_M(pts_2d, pts_3d, num_calibration_pts=12,
                          num_test_pts=4, iterations=10)
    M_16, res_16 = best_M(pts_2d, pts_3d, num_calibration_pts=16,
                          num_test_pts=4, iterations=10)
    results_dict = OrderedDict([('res_8', res_8), ('M_8', M_8.flatten()),
                                ('res_12', res_12), ('M_12', M_12.flatten()),
                                ('res_16', res_16), ('M_16', M_16.flatten())])
    f = open('output/ps3-1-b.txt', 'w')
    for key in results_dict:
        f.write('%s: %s\n'%(key, results_dict[key]))
    f.close()
    print('residual [8]: %.5f, [12]: %.5f, [16]: %.5f\n'%(res_8, res_12, res_16))
    residuals = (res_8, res_12, res_16)
    Ms = (M_8, M_12, M_16)
    res, M = min((res, M) for (res, M) in zip(residuals, Ms))
    return M, res

def ps3_1_c():
    # estimate camera center position in the 3D world coordinates
    M,_ = ps3_1_b()
    Q = M[:, :3]
    m4 = M[:, 3]
    C = np.dot(-np.linalg.inv(Q), m4)
    print('Center of Camera = %s'%C)

def ps3_2_a(printing=True):
    # estimate the Fundamental Matrix between pic_a and pic_b
    pts_a = load_file('input/pts2d-pic_a.txt')
    pts_b = load_file('input/pts2d-pic_b.txt')
    F = least_squares_F_solver(pts_a, pts_b)
    if printing:
        print('Fundametal Matrix with Rank=3: \n%s'%F)
    return F

def ps3_2_b(printing=True):
    F = ps3_2_a(printing=False)
    # reduce the rank of F from 3 to 2
    U,S,V = np.linalg.svd(F)
    S[-1] = 0
    S = np.diag(S)
    F = np.dot(np.dot(U,S), V)
    if printing:
        print('Fundametal Matrix with Rank=2: \n%s'%F)
    return F

def ps3_2_c():
    # load images
    img_a = cv2.imread('input/pic_a.jpg', cv2.IMREAD_COLOR)
    img_b = cv2.imread('input/pic_b.jpg', cv2.IMREAD_COLOR)
    # load points
    pts_a = np.array(load_file('input/pts2d-pic_a.txt'))
    pts_b = np.array(load_file('input/pts2d-pic_b.txt'))
    # convert pts to homogenious coordinates
    pts_a = np.column_stack((pts_a, np.ones(pts_a.shape[0])))
    pts_b = np.column_stack((pts_b, np.ones(pts_a.shape[0])))
    # estimate fundamental matrix
    F = ps3_2_b(printing=False)
    #  F,_ = cv2.findFundamentalMat(pts_a, pts_b, method=cv2.FM_8POINT)
    #  find the epipolar lines for each point in both images
    eplines_a = np.dot(F.T, pts_b.T).T
    eplines_b = np.dot(F, pts_a.T).T
    n, m, _ = img_a.shape
    line_L = np.cross([0,0,1],[n,0,1])
    line_R = np.cross([0,m,1],[n,m,1])
    for line_a, line_b in zip(eplines_a, eplines_b):
        P_a_L = np.cross(line_a, line_L)#.astype(int)
        P_a_R = np.cross(line_a, line_R)#.astype(int)
        P_a_L = (P_a_L[:2] / P_a_R[2]).astype(int)
        P_a_R = (P_a_R[:2] / P_a_R[2]).astype(int)
        cv2.line(img_a, tuple(P_a_L[:2]), tuple(P_a_R[:2]), (0,255,0), thickness=2)
        P_b_L = np.cross(line_b, line_L)
        P_b_R = np.cross(line_b, line_R)
        P_b_L = (P_b_L[:2] / P_b_R[2]).astype(int)
        P_b_R = (P_b_R[:2] / P_b_R[2]).astype(int)
        cv2.line(img_b, tuple(P_b_L[:2]), tuple(P_b_R[:2]), (0,255,0), thickness=2)
    cv2.imshow('', np.hstack((img_a,img_b))); cv2.waitKey(0); cv2.destroyAllWindows()
    # save the images with the highlighted epipolar lines
    cv2.imwrite('output/ps3-2-c-1.png', img_a)
    cv2.imwrite('output/ps3-2-c-2.png', img_b)
    print('Images with highlighted epipolar lines saved successfully!')

def ps3_2_d():
    # load points
    pts_a = np.array(load_file('input/pts2d-pic_a.txt'))
    pts_b = np.array(load_file('input/pts2d-pic_b.txt'))
    # Calculate normalization matrices
    m_a = np.mean(pts_a, axis=0)
    m_b = np.mean(pts_b, axis=0)
    pts_a_temp = np.subtract(pts_a, m_a[None,:])
    pts_b_temp = np.subtract(pts_b, m_b[None,:])
    s_a = 1 / np.abs(np.std(pts_a_temp, axis=0)).max()
    s_b = 1 / np.abs(np.std(pts_b_temp, axis=0)).max()
    S_a = np.diag([s_a, s_a, 1])
    S_b = np.diag([s_b, s_b, 1])
    C_a = np.array([[1, 0, -m_a[0]],[0, 1, -m_a[1]],[0, 0, 1]])
    C_b = np.array([[1, 0, -m_b[0]],[0, 1, -m_b[1]],[0, 0, 1]])
    T_a = np.dot(S_a, C_a)
    T_b = np.dot(S_b, C_b)
    # convert points to homogenious coordinates
    pts_a = np.column_stack((pts_a_temp, np.ones(pts_a.shape[0])))
    pts_b = np.column_stack((pts_b_temp, np.ones(pts_b.shape[0])))
    # Normalize the points (20x3)*(3x3)
    pts_a_norm = np.dot(pts_a, T_a)
    pts_b_norm = np.dot(pts_b, T_b)
    # Estimate fundamental matrix
    F = least_squares_F_solver(pts_a_norm, pts_b_norm)
    # Convert the fundamental matrix to Rank=2
    U,S,V = np.linalg.svd(F)
    S[-1] = 0
    S = np.diag(S)
    F = np.dot(np.dot(U,S), V)
    return T_a, T_b, F

def ps3_2_e():
    # load images
    img_a = cv2.imread('input/pic_a.jpg', cv2.IMREAD_COLOR)
    img_b = cv2.imread('input/pic_b.jpg', cv2.IMREAD_COLOR)
    # load points
    pts_a = np.array(load_file('input/pts2d-pic_a.txt'))
    pts_b = np.array(load_file('input/pts2d-pic_b.txt'))
    # convert pts to homogenious coordinates
    pts_a = np.column_stack((pts_a, np.ones(pts_a.shape[0])))
    pts_b = np.column_stack((pts_b, np.ones(pts_b.shape[0])))
    # Estimate fundamental matrix
    T_a, T_b, F = ps3_2_d()
    # create better fundamental matrix
    F = np.dot(T_b.T, np.dot(F, T_a))
    #  find the epipolar lines for each point in both images
    eplines_a = np.dot(F.T, pts_b.T).T
    eplines_b = np.dot(F, pts_a.T).T
    n, m, _ = img_a.shape
    line_L = np.cross([0,0,1],[n,0,1])
    line_R = np.cross([0,m,1],[n,m,1])
    # draw the epipolar lines on the images
    for line_a, line_b in zip(eplines_a, eplines_b):
        P_a_L = np.cross(line_a, line_L)
        P_a_R = np.cross(line_a, line_R)
        P_a_L = (P_a_L[:2] / P_a_R[2]).astype(int)
        P_a_R = (P_a_R[:2] / P_a_R[2]).astype(int)
        cv2.line(img_a, tuple(P_a_L[:2]), tuple(P_a_R[:2]), (0,255,0), thickness=2)
        P_b_L = np.cross(line_b, line_L)
        P_b_R = np.cross(line_b, line_R)
        P_b_L = (P_b_L[:2] / P_b_R[2]).astype(int)
        P_b_R = (P_b_R[:2] / P_b_R[2]).astype(int)
        cv2.line(img_b, tuple(P_b_L[:2]), tuple(P_b_R[:2]), (0,255,0), thickness=2)
    # save the images with the highlighted epipolar lines
    cv2.imwrite('output/ps3-2-e-1.png', img_a)
    cv2.imwrite('output/ps3-2-e-2.png', img_b)
    print('Images with highlighted epipolar lines saved successfully!')
    print('Fundametal Matrix F=\n%s'%F)
    cv2.imshow('', np.hstack((img_a,img_b))); cv2.waitKey(0); cv2.destroyAllWindows()

ps3_list = OrderedDict([('1a', ps3_1_a), ('1b', ps3_1_b), ('1c', ps3_1_c),
                        ('2a', ps3_2_a), ('2b', ps3_2_b), ('2c', ps3_2_c),
                        ('2d', ps3_2_d), ('2e', ps3_2_e)])

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] in ps3_list:
            print('Executing task %s\n=================='%sys.argv[1])
            ps3_list[sys.argv[1]]()
        else:
            print('Give argument from {1a,1b,2a,2b,2c,2d} for the corresponding task.')
    else:
        print('Executing all tasks:\n')
        for idx in range(len(ps3_list)):
            print('Executing task: %s\n=================='%ps3_list.keys()[idx])
            ps3_list.values()[idx]()


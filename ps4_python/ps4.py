# ps4
import cv2
import numpy as np
import sys
from collections import OrderedDict
import random
from grad_utils import *
from harris_corners import *
from get_keypoints import *
from get_matches import *
from draw_keypoints import *
from ransac_trans import *
from ransac_sim import *
from ransac_sim_affine import *

imgs = ['transA.jpg', 'transB.jpg', 'simA.jpg', 'simB.jpg']

# Harris Corners
# ==============
def ps4_1_a():
    images = imgs[0:3:2]
    for idx, img in enumerate(images):
        img = cv2.imread('input/'+img, cv2.IMREAD_GRAYSCALE)
        k_sobel = 3; k_gauss = 3; s_gauss = 0
        # calculate the X and Y gradients of the two images using the above filter
        img_grad_x = calc_grad_x(img, k_sobel, k_gauss, s_gauss, norm=True)
        img_grad_y = calc_grad_y(img, k_sobel, k_gauss, s_gauss, norm=True)
        # save the gradient pair
        cv2.imwrite('output/ps4-1-a-'+str(idx)+'.png', np.hstack((img_grad_x,
                                                                  img_grad_y)))
    print('Finished calculating and saved the gradients of the images!')
        #  cv2.imshow('', np.hstack((img_grad_x, img_grad_y)))
        # cv2.waitKey(0); cv2.destroyAllWindows()

def ps4_1_b():
    for idx, img_name in enumerate(imgs):
        # read image form file
        img = cv2.imread('input/'+img_name, cv2.IMREAD_GRAYSCALE)
        # calculate harris values for image
        Rs = harris_values(img, window_size=3, harris_scoring=0.04, norm=True)
        # save harris values image
        cv2.imwrite('output/ps4-1-b-'+str(idx+1)+'.png', Rs)
    print('Finished and saved all harris value images to files.')



def ps4_1_c():
    for idx, img_name in enumerate(imgs):
        # read image form file
        img = cv2.imread('input/'+img_name, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        #  corners = cv2.cornerHarris(gray,2,3,0.06) > 0.001*corners.max()
        corners = harris_corners(img, window_size=3, harris_scoring=0.04,
                                 threshold=1e-3, nms_size=5)
        img[corners > 0] = [0, 0, 255]
        cv2.imwrite('output/ps4-1-c-'+str(idx+1)+'.png', img)
        #  cv2.imshow('',img); cv2.waitKey(0); cv2.destroyAllWindows()
    print('Finished harris corner detection and saved new images!')

# SIFT Features
# =============
def ps4_2_a():
    for idx, img_name in enumerate(imgs):
        img = cv2.imread('input/'+img_name, cv2.IMREAD_COLOR)
        img, keypoints = get_keypoints(img)
        cv2.imwrite('output/ps4-2-a-'+str(idx+1)+'.png', img)
    print('Finished keypoint detection and saved drawn images!')


def ps4_2_b():
    for idx in range(len(imgs)/2):
        img1 = cv2.imread('input/'+imgs[2*idx], cv2.IMREAD_COLOR)
        img2 = cv2.imread('input/'+imgs[2*idx+1], cv2.IMREAD_COLOR)
        matching = draw_keypoints(img1, img2)
        cv2.imwrite('output/ps4-2-b-'+str(idx+1)+'.png', matching)
    print('Finished keypoint detection and saved matched keypoints!')

# RANSAC
# ======
def ps4_3_a():
    transA = cv2.imread('input/transA.jpg', cv2.IMREAD_COLOR)
    transB = cv2.imread('input/transB.jpg', cv2.IMREAD_COLOR)
    matching = ransac_trans(transA, transB)
    cv2.imwrite('output/ps4-3-a-1.png', matching)

def ps4_3_b():
    simA = cv2.imread('input/simA.jpg', cv2.IMREAD_COLOR)
    simB = cv2.imread('input/simB.jpg', cv2.IMREAD_COLOR)
    matching,_ = ransac_sim(simA, simB)
    cv2.imwrite('output/ps4-3-b-1.png', matching)

def ps4_3_c():
    simA = cv2.imread('input/simA.jpg', cv2.IMREAD_COLOR)
    simB = cv2.imread('input/simB.jpg', cv2.IMREAD_COLOR)
    matching,_ = ransac_sim_affine(simA, simB)
    cv2.imwrite('output/ps4-3-c-1.png', matching)

def ps4_3_d():
    simA = cv2.imread('input/simA.jpg', cv2.IMREAD_COLOR)
    simB = cv2.imread('input/simB.jpg', cv2.IMREAD_COLOR)
    _,transform = ransac_sim(simA, simB)
    if len(transform) == 0:
        print('Failed to calculate affine transform!')
        return
    warpedB = cv2.warpAffine(simB, transform, simB.shape[1::-1],
                             flags=cv2.WARP_INVERSE_MAP)
    blend = warpedB * 0.5
    blend[:simA.shape[0], :simA.shape[1]] += simA * 0.5
    blend = cv2.normalize(blend, blend, alpha=0, beta=255,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('', blend)
    cv2.waitKey(0); cv2.destroyAllWindows()
    cv2.imwrite('output/ps4-3-d-1.png', blend)

def ps4_3_e():
    simA = cv2.imread('input/simA.jpg', cv2.IMREAD_COLOR)
    simB = cv2.imread('input/simB.jpg', cv2.IMREAD_COLOR)
    _,transform = ransac_sim_affine(simA, simB)
    if len(transform) == 0:
        print('Failed to calculate affine transform!')
        return
    warpedB = cv2.warpAffine(simB, transform, simB.shape[1::-1],
                            flags=cv2.WARP_INVERSE_MAP)
    blend = warpedB * 0.5
    blend[:simA.shape[0], :simA.shape[1]] += simA * 0.5
    blend = cv2.normalize(blend, blend, alpha=0, beta=255,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('', blend)
    cv2.waitKey(0); cv2.destroyAllWindows()
    cv2.imwrite('output/ps4-3-e-1.png', blend)

ps4_list = OrderedDict([('1a', ps4_1_a), ('1b', ps4_1_b), ('1c', ps4_1_c),
                        ('2a', ps4_2_a), ('2b', ps4_2_b), ('3a', ps4_3_a),
                        ('3b', ps4_3_b), ('3c', ps4_3_c), ('3d', ps4_3_d),
                        ('3e', ps4_3_e)])

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] in ps4_list:
            print('\nExecuting task %s\n=================='%sys.argv[1])
            ps4_list[sys.argv[1]]()
        else:
            print('\nGive argument from list {1a,1b,2a,2b,3a,3b,3c,3d}\
                  for the corresponding task.')
    else:
        print('\n * Executing all tasks: * \n')
        for idx in range(len(ps4_list)):
            print('\nExecuting task: %s\n=================='%
                  ps4_list.keys()[idx])
            ps4_list.values()[idx]()

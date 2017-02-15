# ps4
import cv2
import numpy as np
import sys
from collections import OrderedDict
from calc_grad import *
from harris_values import *
from harris_corners import *

imgs = ['transA.jpg', 'transB.jpg', 'simA.jpg', 'simB.jpg']

# Harris Corners
# ==============
def ps4_1_a():
    img1 = cv2.imread('input/transA.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('input/simA.jpg', cv2.IMREAD_GRAYSCALE)
    # get an analytic derivative of a gaussian filter
    k_sobel = 3
    k_gauss = 3
    s_gauss = 0
    # calculate the X and Y gradients of the two images using the above filter
    img1_grad_x = calc_grad_x(img1, k_sobel, k_gauss, s_gauss, norm=True)
    img1_grad_y = calc_grad_y(img1, k_sobel, k_gauss, s_gauss, norm=True)
    img2_grad_x = calc_grad_x(img2, k_sobel, k_gauss, s_gauss, norm=True)
    img2_grad_y = calc_grad_y(img2, k_sobel, k_gauss, s_gauss, norm=True)
    # save gradient pairs
    cv2.imwrite('output/ps4-1-a-1.png', np.hstack((img1_grad_x, img1_grad_y)))
    cv2.imwrite('output/ps4-1-a-2.png', np.hstack((img2_grad_x, img2_grad_y)))
    # display gradients X,Y of img1 and img2
    cv2.imshow('', np.hstack((img1_grad_x,img1_grad_y))); cv2.waitKey(0);
    cv2.imshow('', np.hstack((img2_grad_x,img2_grad_y))); cv2.waitKey(0);
    cv2.destroyAllWindows()

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
        corners = cv2.normalize(corners, corners, alpha=0, beta=255,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #  cv2.imshow('',img); cv2.waitKey(0); cv2.destroyAllWindows()
    print('Finished harris corner detection and saved new images!')

# SIFT Features
# =============
def ps4_2_a():
    for idx, img_name in enumerate(imgs):
        # read image form file
    idx = 0; img_name='simA.jpg'
    img = cv2.imread('input/'+img_name, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.cornerHarris(gray,2,3,0.06) > 0.001*corners.max()
    img[corners > 0] = [0, 0, 255]
    cv2.imwrite('output/ps4-1-c-'+str(idx+1)+'.png', img)
    corners = cv2.normalize(corners, corners, alpha=0, beta=255,
                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #  cv2.imshow('',img); cv2.waitKey(0); cv2.destroyAllWindows()


def ps4_2_b():
    pass
# RANSAC
# ======
def ps4_3_a():
    pass
def ps4_3_b():
    pass
def ps4_3_c():
    pass
def ps4_3_d():
    pass
def ps4_3_e():
    pass

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

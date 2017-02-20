# ps5
import cv2
import numpy as np
import sys
from collections import OrderedDict
import matplotlib.pyplot as plt
from lk_optic_flow import *
from vis_optic_flow import *
from gl_pyramids import *
from flow_to_map import *
from plot_n_save import *
from backwarp import *
from single_level_lk import *

'''
Optic Flow: apparent motion of objects or surfaces
'''

# Lucas Kanade Optic Flow
# =======================
def ps5_1_a():
    fr1 = cv2.imread('input/TestSeq/Shift0.png', cv2.IMREAD_GRAYSCALE)
    fr2 = cv2.imread('input/TestSeq/ShiftR2.png', cv2.IMREAD_GRAYSCALE)
    fr3 = cv2.imread('input/TestSeq/ShiftR5U5.png', cv2.IMREAD_GRAYSCALE)
    #  fr1 = cv2.imread('input/DataSeq2/0.png', cv2.IMREAD_GRAYSCALE)
    #  fr2 = cv2.imread('input/DataSeq2/1.png', cv2.IMREAD_GRAYSCALE)
    #  fr3 = cv2.imread('input/DataSeq2/2.png', cv2.IMREAD_GRAYSCALE)
    #  flow12 = cv2.calcOpticalFlowFarneback(fr2, fr1, None,
                                          #  0.5, 3, 15, 3, 5, 1.2, 0)
    flow12 = lk_optic_flow(fr2, fr1, 15)
    flow13 = lk_optic_flow(fr1, fr3, 15)
    vis_optic_flow_arrows(fr1, flow12, 'output/ps5-1-a-1.png', show=False)
    vis_optic_flow_arrows(fr1, flow13, 'output/ps5-1-a-2.png', show=False)
    #  cv2.imwrite('output/lala/flow1.jpg', vis_optic_flow(flow12))
    #  cv2.imwrite('output/lala/flow2.jpg', vis_optic_flow(flow13))



def ps5_1_b():
    fr1 = cv2.imread('input/TestSeq/Shift0.png', cv2.IMREAD_GRAYSCALE)
    fr2 = cv2.imread('input/TestSeq/ShiftR10.png', cv2.IMREAD_GRAYSCALE)
    fr3 = cv2.imread('input/TestSeq/ShiftR20.png', cv2.IMREAD_GRAYSCALE)
    fr4 = cv2.imread('input/TestSeq/ShiftR40.png', cv2.IMREAD_GRAYSCALE)
    flow12 = lk_optic_flow(fr1, fr2, 15)
    flow13 = lk_optic_flow(fr1, fr3, 15)
    flow14 = lk_optic_flow(fr1, fr4, 15)
    vis_optic_flow_arrows(fr1, flow12, 'output/ps5-1-b-1.png', show=False)
    vis_optic_flow_arrows(fr1, flow13, 'output/ps5-1-b-2.png', show=False)
    vis_optic_flow_arrows(fr1, flow14, 'output/ps5-1-b-3.png', show=False)

# Gaussian and Laplacian Pyramids
# ===============================
def ps5_2_a():
    img = cv2.imread('input/DataSeq1/yos_img_01.jpg', cv2.IMREAD_COLOR)
    gpyr = gaussian_pyramid(img, 4)
    plot_n_save('output/ps5-2-a-1.png', gpyr)

def ps5_2_b():
    img = cv2.imread('input/DataSeq1/yos_img_01.jpg', cv2.IMREAD_COLOR)
    lpyr = laplacian_pyramid(img, 4)
    plot_n_save('output/ps5-2-b-1.png', lpyr)

# TODO Warping by Flow
# ===============
def ps5_3_a_1():
    path = lambda i: 'input/DataSeq1/yos_img_0' + str(i+1) + '.jpg'
    frs = np.array([cv2.imread(path(i), cv2.IMREAD_GRAYSCALE) for i in range(3)])
    # apply LK to the sequence of images using constant level
    flows, warps, diffs = single_level_lk(frs, levels=1, window=15)
    vis_optic_flow_arrows_multi(frs, flows, 'output/ps5-3-a-1.png')
    #  diffs.append(cv2.cvtColor(np.subtract(frs[0], frs[1]), cv2.COLOR_GRAY2BGR))
    #  diffs.append(cv2.cvtColor(np.subtract(frs[1], frs[2]), cv2.COLOR_GRAY2BGR))
    plot_n_save('output/ps5-3-a-2.png', diffs)
    cv2.imwrite('input/DataSeq1/yos_img_1_warped.jpg', warps[0])
    cv2.imwrite('input/DataSeq1/yos_img_2_warped.jpg', warps[1])
    vis = []
    for i, flow in enumerate(flows):
        cv2.imwrite('output/lala/flow1'+str(i)+'.jpg', vis_optic_flow(flow))


def ps5_3_a_2():
    path = lambda i: 'input/DataSeq2/' + str(i) + '.png'
    frs = np.array([cv2.imread(path(i), cv2.IMREAD_GRAYSCALE) for i in range(3)])
    # apply LK to the sequence of images using constant level
    flows, warps, diffs = single_level_lk(frs, levels=2, window=25)
    vis_optic_flow_arrows_multi(frs, flows, 'output/ps5-3-a-3.png')
    plot_n_save('output/ps5-3-a-4.png', diffs)
    vis = []
    for i, flow in enumerate(flows):
        cv2.imwrite('output/lala/flow2'+str(i)+'.png', vis_optic_flow(flow))

# TODO Hierarchical LK Optic Flow
# ==========================
def ps5_4_a():  # HLK on the TestSeq
    pass
def ps5_4_b():  # HLK on the DataSeq1
    pass
def ps5_4_c():  # HLK on the DataSeq2
    pass

# TODO Hierarchical LK to Juggle Sequence
# ==================================
def ps5_5_a():
    pass

ps5_list = OrderedDict([('1a', ps5_1_a), ('1b', ps5_1_b), ('2a', ps5_2_a),
                        ('2b', ps5_2_b), ('3a1', ps5_3_a_1), ('3a2', ps5_3_a_2),
                        ('4a', ps5_4_a), ('4b', ps5_4_b), ('4c', ps5_4_c),
                        ('5a', ps5_5_a)])

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] in ps5_list:
            print('\nExecuting task %s\n=================='%sys.argv[1])
            ps5_list[sys.argv[1]]()
        else:
            print('\nGive argument from list \n%s\nfor the corresponding task.'
                  %ps5_list.keys())
    else:
        print('\n * Executing all tasks: * \n')
        for idx in range(len(ps5_list)):
            print('\nExecuting task: %s\n=================='%
                  ps5_list.keys()[idx])
            ps5_list.values()[idx]()

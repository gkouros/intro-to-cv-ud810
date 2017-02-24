# ps7
import cv2
import numpy as np
import sys
import time
from collections import OrderedDict
from naive_pf_tracker_demo import *
from msl_pf_tracker_demo import *

'''
Problem Set 6: Particle Tracking
'''

videos = ['pres_debate', 'noisy_debate', 'pedestrians']
textfiles = ['pres_debate', 'noisy_debate', 'pedestrians', 'pres_debate_hand']

# 1. Frame-Differenced MHI
# ===========================
def ps7_1_a():
    pass
def ps7_1_b():
    pass

# 2. Recognition using MHIs
# ==========================
def ps7_2_a():
    pass
def ps7_2_b():
    pass


ps7_list = OrderedDict([('1a', ps7_1_a), ('1e', ps7_1_e), ('2a', ps7_2_a),
                        ('2b', ps7_2_b), ('3a', ps7_3_a), ('3b', ps7_3_b),
                        ('4a', ps7_4_a), ('4b', ps7_4_b)])

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] in ps7_list:
            print('Executing task %s'%sys.argv[1])
            ps7_list[sys.argv[1]]()
        else:
            print('\nGive argument from list \n%s\nfor the corresponding task.'
                  %ps7_list.keys())
    else:
        print('\n * Executing all tasks: * \n')
        for idx in range(len(ps7_list)):
            print('Executing task: %s'%
                  ps7_list.keys()[idx])
            ps7_list.values()[idx]()

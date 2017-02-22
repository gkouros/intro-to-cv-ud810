# ps6
import cv2
import numpy as np
import sys
import time
from collections import OrderedDict
from naive_pf_tracker_demo import *

'''
Problem Set 6: Particle Tracking
'''

videos = ['pres_debate', 'noisy_debate', 'pedestrians']

# 1. Particle Filter Tracking
# ================================
def ps6_1_a():
    naive_pf_tracker_demo(videos[0], [28,84,144], 'a', play_video=True,
                         num_particles=100, dimensions=2, control=10, noise=10,
                         sim_std=30)

def ps6_1_e():
    naive_pf_tracker_demo(videos[1], [14,32,46], 'e', play_video=True,
                          num_particles=100, dimensions=2, control=10, noise=10,
                          sim_std=20)

# TODO 2. Appearance Model Update
# ===============================
def ps6_2_a():
    pass

def ps6_2_b():
    pass

# TODO 3. Mean-Shift Lite
# =======================
def ps6_3_a():
    pass
def ps6_3_b():
    pass

# TODO 4. Incorporating More Dynamics
# ===================================
def ps6_4_a():
    pass
def ps6_4_b():
    pass

ps6_list = OrderedDict([('1a', ps6_1_a), ('1e', ps6_1_e), ('2a', ps6_2_a),
                        ('2b', ps6_2_b), ('3a', ps6_3_a), ('3b', ps6_3_b),
                        ('4a', ps6_4_a), ('4b', ps6_4_b)])

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] in ps6_list:
            print('\nExecuting task %s\n=================='%sys.argv[1])
            ps6_list[sys.argv[1]]()
        else:
            print('\nGive argument from list \n%s\nfor the corresponding task.'
                  %ps6_list.keys())
    else:
        print('\n * Executing all tasks: * \n')
        for idx in range(len(ps6_list)):
            print('\nExecuting task: %s\n=================='%
                  ps6_list.keys()[idx])
            ps6_list.values()[idx]()

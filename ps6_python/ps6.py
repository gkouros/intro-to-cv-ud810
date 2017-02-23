# ps6
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

# 1. Particle Filter Tracking
# ================================
def ps6_1_a():
    naive_pf_tracker_demo(videos[0], textfiles[0], [28,84,144], '1-a',
                          play_video=True, num_particles=100, dimensions=2,
                          control=10, sim_std=20, alpha=0)

def ps6_1_e():
    naive_pf_tracker_demo(videos[1], textfiles[1], [14,32,46], '1-e',
                          play_video=True, num_particles=100, dimensions=2,
                          control=10, sim_std=10, alpha=0)

# 2. Appearance Model Update
# ===============================
def ps6_2_a():
    naive_pf_tracker_demo(videos[0], textfiles[3], [15,50,140], '2-a',
                          play_video=True, num_particles=700, dimensions=2,
                          control=10, sim_std=5, alpha=0.1)

def ps6_2_b():
    naive_pf_tracker_demo(videos[1], textfiles[3], [15,50,140], '2-b',
                          play_video=True, num_particles=1000, dimensions=2,
                          control=10, sim_std=2, alpha=0.2)

# 3. Mean-Shift Lite
# =======================
def ps6_3_a():
    msl_pf_tracker_demo(videos[0], textfiles[0], [28,84,144], '3-a',
                        play_video=True, num_particles=1000, dimensions=2,
                        control=5, sim_std=10, alpha=0)
def ps6_3_b():
    msl_pf_tracker_demo(videos[0], textfiles[3], [15,50,140], '3-b',
                        play_video=True, num_particles=1000, dimensions=2,
                        control=10, sim_std=1, alpha=0.1)

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

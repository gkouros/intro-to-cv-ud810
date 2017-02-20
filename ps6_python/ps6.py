# ps6
import cv2
import numpy as np
import sys
from collections import OrderedDict

'''
Problem Set 6: Particle Tracking
'''

# TODO 1. Particle Filter Tracking
# ================================
def ps6_1_a():
    pass
def ps6_1_b():
    pass
def ps6_1_c():
    pass
def ps6_1_d():
    pass
def ps6_1_e():
    pass

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

ps6_list = OrderedDict([('1a', ps6_1_a), ('1b', ps6_1_b), ('1c', ps6_1_c),
                        ('1d', ps6_1_d), ('1e', ps6_1_e), ('2a', ps6_2_a),
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

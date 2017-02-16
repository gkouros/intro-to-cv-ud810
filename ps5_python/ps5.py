# ps4
import cv2
import numpy as np
import sys
from collections import OrderedDict
import random

# Lucas Kanade Optic Flow
# =======================
def ps5_1_a():
    pass
def ps5_1_b():
    pass

# Gaussian and Laplacian Pyramids
# ===============================
def ps5_2_a():
    pass
def ps5_2_b():
    pass

# Warping by Flow
# ===============
def ps5_3_a():
    pass

# Hierarchical LK Optic Flow
# ==========================
def ps5_4_a():
    pass
def ps5_4_b():
    pass
def ps5_4_c():
    pass

# Hierarchical LK to Juggle Sequence
# ==================================
def ps5_5_a():
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

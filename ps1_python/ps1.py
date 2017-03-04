#!/usr/bin/env python2

import time
from collections import OrderedDict
import sys
from ps1_python import *

'''
Problem Set 1: Edges and Lines
==============================
1: Edge image
2: Hough transform for lines
3: Hough transform for lines on noisy image
4: Hough transform for lines on a more complex image
5: Hough Transform for Circle Detection
6: Apply line detection on image with cluttering
7: Apply Hough circle detection on the cluttered image
8: Apply line and circle detection on the distorted cluttered image
'''

ps1_list = OrderedDict([('1', ps1_1.ps1_1), ('2', ps1_2.ps1_2),
                        ('3', ps1_3.ps1_3), ('4', ps1_4.ps1_4),
                        ('5', ps1_5.ps1_5), ('6', ps1_6.ps1_6),
                        ('7', ps1_7.ps1_7), ('8', ps1_8.ps1_8)])

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] in ps1_list:
            print('Executing task %s'%sys.argv[1])
            ps1_list[sys.argv[1]]()
        else:
            print('\nGive argument from list \n%s\nfor the corresponding task.'
                  %ps1_list.keys())
    else:
        print('\n * Executing all tasks: * \n')
        for idx in range(len(ps1_list)):
            print('Executing task: %s'%
                  ps1_list.keys()[idx])
            ps1_list.values()[idx]()

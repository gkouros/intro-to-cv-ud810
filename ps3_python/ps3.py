# ps3
import sys
import numpy as np
import cv2

def ps3_1():
    pass
def ps3_2():
    pass

ps3_list = [ps3_1, ps3_2]

if len(sys.argv) == 2:
    if int(sys.argv[1]) in range(1, len(ps3_list)):
        print('Executing task %d'%(int(sys.argv[1])))
        ps3_list[int(sys.argv[1])-1]()
    else:
        print('Give argument in range [1,2] for the corresponding tasks')
else:
    print('Executing all tasks:')
    for idx, ps in enumerate(ps3_list):
        print('Executing task: %d'%(idx+1))
        ps()


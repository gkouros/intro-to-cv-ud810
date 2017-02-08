#!/usr/bin/env python3

import time
import ps1_1, ps1_2, ps1_3, ps1_4, ps1_5, ps1_6, ps1_7, ps1_8

start_time = time.time()

# 1: Edge image
ps1_1.ps1_1_fun()
# 2: Hough transform for lines
ps1_2.ps1_2_fun()
# 3: Hough transform for lines on noisy image
ps1_3.ps1_3_fun()
#  4: Hough transform for lines on a more complex image
ps1_4.ps1_4_fun()
# 5: Hough Transform for Circle Detection
ps1_5.ps1_5_fun()
#  6: Apply line detection on image with cluttering
ps1_6.ps1_6_fun()
# 7: Apply Hough circle detection on the cluttered image
ps1_7.ps1_7_fun()
# 8: Apply line and circle detection on the distorted cluttered image
ps1_8.ps1_8_fun()

print('Finished in %.1f s'%(time.time() - start_time))

import time
import numpy as np

start = time.time()
a = np.zeros(1000)
for i in range(1000):
    a[i] = i
period = time.time()-start

start2 = time.time()
a = [i for i in range(1000)]
period2 = time.time()-start2
print(period, period2)
print('relation: %f'%float(period/period2))

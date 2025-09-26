import numpy as np
for p in ["data/xb.npy","data/xq.npy","data/gt.npy"]:
    a = np.load(p, mmap_mode='r')
    print(p, a.shape, a.dtype)

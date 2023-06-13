import numpy as np
import cv2

# This is OK
s = 11 # 128, 256 the same results
img = np.ones((s, s, 9), dtype=np.float)
img[2:5,2:5, :] = 2
print(img[:,:,1])
M = np.float32([[1, 0, -5], [0, 1, -5]])
shifted = cv2.warpAffine(img, M, (s, s))
print(shifted[:,:,1])

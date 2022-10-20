from glob import glob
f = glob('/home/qinyu.zhao/datasets/synthesis_dataset/*.*')
print(len(f))

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(f[np.random.randint(len(f))])
img = img[:, :, ::-1]
plt.imshow(img)
# import required libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt

# read two input images as grayscale images
imgL = cv2.imread('L.png',0)
imgR = cv2.imread('R.png',0)

# Initiate and StereoBM object
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# compute the disparity map
disparity = stereo.compute(imgL,imgR)
cv2.imwrite('depth_map.png', disparity)

disp_v2 = cv2.imread('depth_map.png')
disp_v2 = cv2.applyColorMap(disp_v2, cv2.COLORMAP_JET)

plt.imshow(disparity,'gray')
plt.imshow(disp_v2)

cv2.imwrite('depth_map_coloured.png', disp_v2)

plt.show()
# disparity.shape
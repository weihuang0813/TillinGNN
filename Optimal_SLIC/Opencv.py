import numpy as np

# from fast_slic import Slic
# from PIL import Image

# with Image.open("mu.png") as f:
#    image = np.array(f)

import cv2; 
image = cv2.imread('mu.png')
image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)   # You can convert the image to CIELAB space if you need.

print(image)
cv2.imshow("assignment", image)
cv2.waitKey(0)


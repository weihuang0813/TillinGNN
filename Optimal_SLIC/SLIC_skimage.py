from skimage.segmentation import slic,mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
# import numpy as np
#
# np.set_printoptions(threshold=np.inf)

img = io.imread("fish.jpg")


segments = slic(img, n_segments=60, compactness=10)
out=mark_boundaries(img,segments)
# print(segments)
plt.subplot(121)
plt.title("n_segments=60")
plt.imshow(out)

segments2 = slic(img, n_segments=300, compactness=10)
out2=mark_boundaries(img,segments2)
plt.subplot(122)
plt.title("n_segments=300")
plt.imshow(out2)

plt.show()
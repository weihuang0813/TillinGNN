import cv2; 
import numpy as np

from fast_slic import Slic
from PIL import Image
from shapely.geometry import Point, Polygon


# with Image.open("mu.png") as f:
#    image = np.array(f)

image = cv2.imread('dog1-01.png')
height, width = image.shape[:2]
shape = (height, width, 3) # y, x, RGB
origin_img = np.full(shape, 255).astype(np.uint8)

image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)   # You can convert the image to CIELAB space if you need.

slic = Slic(num_components=500, compactness=50,subsample_stride=1,debug_mode=True)
assignment = slic.iterate(image) # Cluster Map

# print(type(slic.slic_model.clusters))
temp = slic.slic_model.clusters # The cluster information of superpixels.

# print(assignment)
# print(assignment.shape)
# print(temp)

for i in range(len(temp)):
   print(temp[i])
   y_ray = (int)(temp[i]["yx"][0])
   x_ray = (int)(temp[i]["yx"][1])
   p1 = Point(x_ray,y_ray)
   print("p1 = ",p1)


   lab = temp[i]["color"]
   # rgb = Lab2RGB(lab)
   # print(rgb)
   rgb = cv2.cvtColor( np.uint8([[lab]] ), cv2.COLOR_LAB2RGB)[0][0]
   color = tuple ([int(x) for x in rgb])
   print(color)
   # cv2.rectangle(origin_img, (x_ray-10,y_ray-14), (x_ray+10,y_ray+14), color, -1)
   cv2.circle(origin_img, (x_ray,y_ray), 2, color, 5)
   # cv2.imshow("origin_img", origin_img)
   # cv2.waitKey(0)
   # print(image[y_ray,x_ray]) 


print(len(slic.slic_model.clusters))

cv2.imwrite('dog1-01_slic.png', origin_img)
cv2.imshow("origin_img", origin_img)
cv2.imshow("image", image)
cv2.waitKey(0)
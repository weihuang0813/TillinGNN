import cv2  
import numpy as np
  
img = cv2.imread("/media/cglab/CEDCACB9DCAC9D69/TilinGNN-test/silhouette/alien2.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray,128,255,cv2.THRESH_BINARY_INV)
#ret, binary = cv2.threshold(gray,84,255,cv2.THRESH_TOZERO_INV)  
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
cv2.drawContours(img,contours,0,(0,0,255),3)  
#'''
#cv2.imshow('My Image', gray)
cv2.imshow('My Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#'''
f = open("./silhouette/alien2.txt", "w")
for level in range(len(contours)):
    #if len(contours) > 1 and level == 0:
    #    continue
    for i in range(len(contours[level])):
        if i == len(contours[level]) - 1:
            f.write(str(contours[level][i][0][0]) + " " + str(contours[level][i][0][1]))
        else:
            f.write(str(contours[level][i][0][0]) + " " + str(contours[level][i][0][1])+",")
    f.write("\n")
f.close()
#'''
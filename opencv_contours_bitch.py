import numpy as np
import cv2

for i in range(4):
    img = cv2.imread("/media/cglab/CEDCACB9DCAC9D69/TilinGNN-test/silhouette/img_cap" + str(i) + ".jpg")
    img = cv2.resize(img, None,fx=1, fy=1, interpolation = cv2.INTER_CUBIC)
    print("LOAD....img_cap" + str(i) )
    # get a blank canvas for drawing contour on and convert img to grayscale
    canvas = np.zeros(img.shape, np.uint8)
    img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # filter out small lines between counties
    kernel = np.ones((5,5),np.float32)/25
    img2gray = cv2.filter2D(img2gray,-1,kernel)

    # threshold the image and extract contours
    ret,thresh = cv2.threshold(img2gray,245,255,cv2.THRESH_BINARY_INV) ## 顏色二值化（黑色白色） 目的將圖片顏色落差太大的區分開成黑色白色
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # find the main island (biggest area)
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)
    for cont in contours: ### convex 演算法
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)

    cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 3)

    f = open("./silhouette/img_cap"+str(i)+".txt", "w")
    for j in range(len(cnt)):
        if j == len(cnt) - 1:
            f.write(str(cnt[j][0][0]) + " " + str(cnt[j][0][1]))
        else:
            f.write(str(cnt[j][0][0]) + " " + str(cnt[j][0][1])+",")
    f.close()


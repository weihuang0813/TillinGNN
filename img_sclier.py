import numpy as np
import cv2

# load image and shrink - it's massive
img = cv2.imread("/media/cglab/CEDCACB9DCAC9D69/TilinGNN-test/silhouette/dog4.png")
# img = cv2.resize(img, None,fx=1, fy=1, interpolation = cv2.INTER_CUBIC)
img_cap = []
step_x = (int)(img.shape[0]/2)
step_y = (int)(img.shape[1]/2)
for row in range(0, img.shape[0], step_x):
    for col in range(0, img.shape[1], step_y):
        if(((row + step_x) <= img.shape[0]) and ((col + step_y) <= img.shape[1])):
            ltop = (row, col)
            rtbm = (row + step_x, col + step_y)
            img_cap.append(img[ltop[1]:rtbm[1], ltop[0]: rtbm[0]])
cv2.imshow("image",img)
for i in range(4):
    img_name = "img_cap"+ str(i)
    cv2.imshow(img_name,img_cap[i])
    winname = img_name
    cv2.moveWindow(winname, 40 + i * 400,200)
    cv2.imwrite("./silhouette/" + img_name + ".jpg", img_cap[i])

# cv2.imshow('canvas',canvas)
# f = open("./silhouette/dog4.txt", "w")
# for i in range(len(cnt)):
#     if i == len(cnt) - 1:
#         f.write(str(cnt[i][0][0]) + " " + str(cnt[i][0][1]))
#     else:
#         f.write(str(cnt[i][0][0]) + " " + str(cnt[i][0][1])+",")
# f.close()


k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
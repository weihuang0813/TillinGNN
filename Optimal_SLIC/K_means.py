import cv2
import numpy as np
from sklearn.cluster import KMeans

# 读入图像
img = cv2.imread('dog1.png')

# 将图像转换为 LAB 空间
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# 建立一個二值化的區域 mask
mask = np.zeros(img.shape[:2], dtype=np.uint8)
pts = np.array([[100, 100], [200, 100], [150, 200]])
cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1)

masked_image = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('Cropped Image', masked_image)

# 使用 boolean indexing 篩選出區域內的像素值
region_pixels = lab[mask == 255]

# # 获取切割范围内的颜色
# cropped_image = lab[280:480, 330:510]
# # colors = img[mask > 0].reshape(-1, 3)
# colors = cropped_image.reshape(-1, 3)

# 使用 k-means 聚合颜色
# kmeans = KMeans(n_clusters=3, random_state=0).fit(colors)
kmeans = KMeans(n_clusters=1, random_state=0).fit(region_pixels)
# kmeansImg = (kmeans.labels_).reshape(cropped_image.shape[0], cropped_image.shape[1])
cluster_centers = kmeans.cluster_centers_

# 返回n個聚合過後的顏色值
bgr_pixel = cv2.cvtColor(np.uint8([cluster_centers]), cv2.COLOR_LAB2BGR)[0]
print("bgr_pixel = ",bgr_pixel)

# 将图像颜色设置为聚类中心颜色
# new_img = np.zeros_like(img)
# new_img = np.zeros_like(cropped_image)
# for i, color in enumerate(kmeans.cluster_centers_):
#     # print(i) # 第幾分類標籤
#     # print(len(kmeansImg)) # [0 0 0 ... 0 0 0]
#     # new_img[mask > 0][kmeansImg == i] = color
#     new_img[kmeansImg == i] = color.astype(int)

# 建立一個黑色畫布
canvas = np.zeros((300, 300, 3), dtype=np.uint8)

# 使用 cv2.rectangle 函數畫正方形
cv2.rectangle(canvas, (0, 0), (300, 300), (int(bgr_pixel[0][0]), int(bgr_pixel[0][1]), int(bgr_pixel[0][2])), thickness=-1)

# 将图像转换回 BGR 空间
# new_img = cv2.cvtColor(new_img, cv2.COLOR_LAB2BGR)

# 保存处理后的图像
# cv2.imwrite('new_dog1.png', new_img)
# cv2.imshow("new_img", new_img)
cv2.imshow("img", img)
cv2.imshow("Square", canvas)
cv2.waitKey(0)
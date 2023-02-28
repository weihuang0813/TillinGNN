import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances_argmin_min

# 读入图像
img = cv2.imread('wolf.png')

# 将图像转换为 LAB 空间
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# 建立一個二值化的區域 mask
mask = np.zeros(img.shape[:2], dtype=np.uint8)
pts = np.array([[150, 100], [200, 100], [200, 150], [150, 150]])
cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1)

masked_image = cv2.bitwise_and(img, img, mask=mask)

# 使用 boolean indexing 篩選出區域內的像素值
region_pixels = lab[mask == 255]                      

# 使用 k-means 聚合颜色
kmeans = KMeans(n_clusters=5, random_state=0, init='k-means++').fit(region_pixels)

# 假设数据为X，聚类数从2到10
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++')
    cluster_labels = kmeans.fit_predict(region_pixels)
    silhouette_avg = silhouette_score(region_pixels, cluster_labels)
    print(f"For n_clusters = {n_clusters}, the average silhouette score is : {silhouette_avg:.3f}")

# print("kmeans.labels_ = ",kmeans.labels_)

# # 計算每個label的Silhouette Score
# silhouette_scores = []
# for label in range(5):
#     silhouette_scores.append(silhouette_score(region_pixels, kmeans.labels_))

# # 找出最大的Silhouette Score所對應的label
# best_label = silhouette_scores.index(max(silhouette_scores))
# print("best_label = ",best_label)

# 计算每个点到其所属质心的距离平方和
distances = pairwise_distances_argmin_min(region_pixels, kmeans.cluster_centers_)
sse = sum(distances[1])

# 选择最小的SSE对应的标签
best_label = kmeans.labels_[distances[0][np.argmin(distances[1])]]

cluster_centers = kmeans.cluster_centers_

# 返回n個聚合過後的顏色值
# 将图像转换回 BGR 空间
# bgr_pixel = np.uint8([cluster_centers])[0] # wothout LAB
bgr_pixel = cv2.cvtColor(np.uint8([cluster_centers]), cv2.COLOR_LAB2BGR)[0]
print("bgr_pixel = ",bgr_pixel)
print("bgr_pixel[best_label] = ",bgr_pixel[best_label])

# 建立一個黑色畫布
canvas = np.zeros((300, 300, 3), dtype=np.uint8)

# 使用 cv2.rectangle 函數畫正方形
cv2.rectangle(canvas, (0, 0), (300, 300), (int(bgr_pixel[best_label][0]), int(bgr_pixel[best_label][1]), int(bgr_pixel[best_label][2])), thickness=-1)

cv2.imshow('Cropped Image', masked_image)
cv2.imshow("img", img)
cv2.imshow("Square", canvas)
cv2.waitKey(0)
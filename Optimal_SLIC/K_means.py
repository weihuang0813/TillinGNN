import cv2
import numpy as np
from sklearn.cluster import KMeans

# 读入图像
img = cv2.imread('fish.jpg')

# 获取切割的范围
# mask = cv2.imread('mask.jpg', 0)
# mask = (mask > 128).astype(np.uint8)

# 将图像转换为 LAB 空间
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# 获取切割范围内的颜色
# colors = img[mask > 0].reshape(-1, 3)
colors = lab.reshape(-1, 3)

# 使用 k-means 聚合颜色
kmeans = KMeans(n_clusters=5, random_state=0).fit(colors)
kmeansImg = (kmeans.labels_).reshape(img.shape[0], img.shape[1])

# 将图像颜色设置为聚类中心颜色
new_img = np.zeros_like(img)
for i, color in enumerate(kmeans.cluster_centers_):
    print(i)
    print(len(kmeansImg)) # [0 0 0 ... 0 0 0]
    # new_img[mask > 0][kmeansImg == i] = color
    new_img[kmeansImg == i] = color.astype(int)


# 将图像转换回 BGR 空间
new_img = cv2.cvtColor(new_img, cv2.COLOR_LAB2BGR)

# 保存处理后的图像
cv2.imwrite('new_fish.jpg', new_img)
cv2.imshow("new_img", new_img)
cv2.waitKey(0)
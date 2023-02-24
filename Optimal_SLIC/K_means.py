import cv2
import numpy as np
from sklearn.cluster import KMeans

# 读入图像
img = cv2.imread('dog1.png')

# 获取切割的范围
# mask = cv2.imread('mask.jpg', 0)
# mask = (mask > 128).astype(np.uint8)

# 将图像转换为 LAB 空间
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


# 定義多邊形的頂點座標
pts = np.array([[150, 100], [200, 100], [200, 150], [150, 150]], np.int32)

# 創建一個空白圖像，大小與原圖片相同
mask = np.zeros_like(img)

# 將多邊形的內部區域填充為白色
cv2.fillPoly(mask, [pts], (255,255,255))

# 將原圖片和填充後的多邊形進行按位與運算
cropped_image = cv2.bitwise_and(img, mask)


# 获取切割范围内的颜色
# cropped_image = lab[80:280, 150:330]
cropped_image = lab[280:480, 330:510]
# colors = img[mask > 0].reshape(-1, 3)
colors = cropped_image.reshape(-1, 3)

# 使用 k-means 聚合颜色
kmeans = KMeans(n_clusters=1, random_state=0).fit(colors)
kmeansImg = (kmeans.labels_).reshape(cropped_image.shape[0], cropped_image.shape[1])

# 将图像颜色设置为聚类中心颜色
# new_img = np.zeros_like(img)
new_img = np.zeros_like(cropped_image)
for i, color in enumerate(kmeans.cluster_centers_):
    print(i)
    print(len(kmeansImg)) # [0 0 0 ... 0 0 0]
    # new_img[mask > 0][kmeansImg == i] = color
    new_img[kmeansImg == i] = color.astype(int)


# 将图像转换回 BGR 空间
new_img = cv2.cvtColor(new_img, cv2.COLOR_LAB2BGR)

# 保存处理后的图像
# cv2.imwrite('new_dog1.png', new_img)
cv2.imshow("new_img", new_img)
cv2.imshow("cropped_image", cropped_image)
cv2.imshow("img", img)
cv2.waitKey(0)
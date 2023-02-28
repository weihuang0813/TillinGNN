import numpy as np
import math
import cv2 
import os
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score, calinski_harabasz_score
from fast_slic import Slic
from shapely.geometry import Point, Polygon
import copy

def calculation_of_transform (file_name, original_contour): #計算從gui介面的輪廓轉換到原圖的輪廓所作的縮放和平移
    trans = [1,0,0]
    coordinate = []
    with open(file_name) as f:
        for line in f.readlines():
            s = line.replace(","," ").split(' ')
            if(len(s)%2 != 0):
                print("coordinate is wrong！!！")
            for i in range(int(len(s)/2)):
                coordinate.append([])
                coordinate[len(coordinate)-1].append(float(s[i*2]))
                coordinate[len(coordinate)-1].append(float(s[i*2+1]))
            #coordinate.append(coordinate[0])
            # print(f"txt : {len(coordinate)}")
            # print(coordinate)
            # print(f"contour : {len(original_contour)}")
            # print(original_contour)
            break #只算外輪廓～忽略內輪廓因此直接break
        '''
    if(len(coordinate) != len(original_contour)):
        print("two counters do not have same length")
        print(f"txt: : {len(coordinate)}")
        print(coordinate)
        print(f"contour : {len(original_contour)}")
        print(original_contour)
    else:
        '''
    for i in range(len(coordinate)):
        m1 = original_contour[i]
        m2 = original_contour[i+1]
        n1 = coordinate[i]
        n2 = coordinate[i+1]
        if(m1[0] == m2[0] or m1[1] == m2[1] or n1[0] == n2[0] or n1[1] == n2[1]):
            continue
        a1 = (n1[0]-n2[0]) / (m1[0]-m2[0])
        a2 = (n1[1]-n2[1]) / (m1[1]-m2[1])
        b1 = n1[0] - a1*m1[0]
        b2 = n2[0] - a1*m2[0]
        c1 = n1[1] - a2*m1[1]
        c2 = n2[1] - a2*m2[1]
        if (round(a1,10)==round(a2,10) and round(b1,10)==round(b2,10) and round(c1,10)==round(c2,10)):
            trans[0] = round(a1,10)
            trans[1] = round(b1,10)
            trans[2] = round(c1,10)
            break
        else:
            continue
            
    return trans

def color_catch_allen_wang(trans, file_name, tiles): #抓取原圖中指定範圍內的所有像素點顏色
    
    img_name = file_name[:-4]
    png = [".jpeg", ".png", ".jpg", ".PNG", ".JPG"]
    find_file = False
    img = []
    for i in range(len(png)):
        if os.path.exists(img_name + png[i]):
            if os.path.isfile(img_name + png[i]):
                img = cv2.imread(img_name  + png[i])
                # image = cv2.imread(img_name + png[i])

                # slic = Slic(num_components=2000, compactness=150,subsample_stride=1,debug_mode=True)
                # assignment = slic.iterate(image) # Cluster Map
                # temp = slic.slic_model.clusters # The cluster information of superpixels.

                find_file = True
    if(find_file == False):
        print("no file")



    h_jump = int(len(img)/100)
    w_jump = int(len(img[0])/100)
    zoom = trans[0]
    move_x = trans[1]
    move_y = trans[2]
    tiles_color = []
    tiles_tmp = [] # tiles 轉換資料型態成 list
    # tiles_tmp.append(list(tiles))
    # print("tiles = ",tiles)
    # print("\n\n\n\n\n\n\n\n")

    tiles_tmp = copy.deepcopy(tiles)
    # print(tiles) # ('yellow', array([[2.5 , 0.9 ],[2.5 , 1.2 ],[2.75, 0.9 ],[2.5 , 0.9 ]]} 鋪磚狀況無顏色
    for i in range(len(tiles_tmp)):
        tiles_tmp[i] = list(tiles_tmp[i])
        # tiles_tmp[i].append("false") # init紀錄有沒有找過
        tiles_tmp[i][1][:,0] = tiles_tmp[i][1][:,0] * zoom + move_x # for why tiles & tiles-tmp 都一起被改了
        tiles_tmp[i][1][:,1] = tiles_tmp[i][1][:,1] * zoom + move_y

    # print("tiles = ",tiles)
    # print("\n\n\n\n\n\n\n\n")
    print("tiles_tmp = ",tiles_tmp)

    for i in range(len(tiles)):
        #print("#######################")
        #print(i)
        xmin = 10000
        ymin = 10000
        xmax = 0
        ymax = 0
        xnow = 0
        ynow = 0
        # tiles_ploygon = []
        # tiles_tmp_color = []
        # test = []

        # # print("tiles[i][1] = ",tiles[i][1]) # tiles[i][1] =  [[2.5  0.6 ] [2.5  0.9 ] [2.75 0.9 ] [2.75 0.6 ] [2.5  0.6 ]]
        # # tiles[i][1] 乘上 zoom + (move_x,move_y)

        # # ppx, ppy = zip(*tiles[i][1])

        # tiles_ploygon.append(tiles_tmp[i][1])
        # tploy = Polygon(tiles_ploygon[0]) # tiles_ploygon = [[[0.0, 0.0], [0.0, 0.3], [0.25, 0.3], [0.25, 0.0], [0.0, 0.0]]]
        # print(tploy.area)

        # for index in range(len(temp)):
        #     # print(temp[index])
        #     y_ray = (int)(temp[index]["yx"][0])
        #     x_ray = (int)(temp[index]["yx"][1])
        #     p1 = Point(x_ray,y_ray)
        #     # print("p1 = ",p1)

        #     if(p1.within(tploy) and tiles_tmp[i][2] == 'false'):
        #         tiles_tmp[i][2] = 'true'
        #         lab = temp[index]["color"]
        #         rgb = cv2.cvtColor( np.uint8([[lab]] ), cv2.COLOR_LAB2RGB)[0][0]
        #         tiles_color.append(rgb)

        #     if(p1.within(tploy)):
        #         lab = temp[index]["color"]
        #         rgb = cv2.cvtColor( np.uint8([[lab]] ), cv2.COLOR_LAB2RGB)[0][0]
        #         rgb = list(rgb)
        #         tiles_tmp_color.append(rgb)

        # for _index in range(len(tiles_tmp_color)):
        #     print("tiles_tmp_color[_index] = ",tiles_tmp_color[_index])
        #     print("tiles_tmp_color.count(tiles_tmp_color[_index]) = ",tiles_tmp_color.count(tiles_tmp_color[_index]) )
        #     if(tiles_tmp_color.count(tiles_tmp_color[_index]) == 1):
        #         test.append(tiles_tmp_color[_index])
        #         break
        #     elif(tiles_tmp_color.count(tiles_tmp_color[_index]) == 2):
        #         test.append(tiles_tmp_color[_index])

        # print("test = ",test[0][0][0][0])

        # print("tiles_tmp_color = ",tiles_tmp_color)
        
        for j in range(len(tiles[i][1])-1):
            if(tiles[i][1][j][0] * zoom + move_x > xmax): xmax = tiles[i][1][j][0] * zoom + move_x
            if(tiles[i][1][j][0] * zoom + move_x < xmin): xmin = tiles[i][1][j][0] * zoom + move_x
            if(tiles[i][1][j][1] * zoom + move_y > ymax): ymax = tiles[i][1][j][1] * zoom + move_y
            if(tiles[i][1][j][1] * zoom + move_y < ymin): ymin = tiles[i][1][j][1] * zoom + move_y
        if(xmin == 10000 or ymin == 10000 or xmax == 0 or ymax == 0):
            print("################")
            print("Bricks transform have wrong !!!")
            print("################")
            break

        # # print("xmax,xmin,ymax,ymin",xmax,xmin,ymax,ymin)
        # '''
        # if(len(tiles[i][1])-1 == 4):
        #     print("???")
        # else:
        #     print("!!!")
        # '''
        bgr_total = []
        xnow = int(round(xmin, 0))
        ynow = int(round(ymin, 0))
        #print(xmin)
        #print(xmax)
        #print(ymin)
        #print(ymax)
        if(xmax - xmin < w_jump):
            w_jump = 1
        if(ymax - ymin < h_jump):
            h_jump = 1
        while(xnow < xmax-1):
            while(ynow < ymax-1):
                #基本磚
                if(len(tiles[i][1])-1 == 4):
                    #抓取圖片（xnow,ynow）位置的bgr值紀錄
                    bgr_total.append(img[ynow][xnow])
                    # print("ynow = ",ynow)
                    # print("xnow = ",xnow)
                #斜磚
                else:
                    ab = [(tiles[i][1][1][0]-tiles[i][1][0][0]) * zoom, (tiles[i][1][1][1]-tiles[i][1][0][1]) * zoom]
                    bc = [(tiles[i][1][2][0]-tiles[i][1][1][0]) * zoom, (tiles[i][1][2][1]-tiles[i][1][1][1]) * zoom]
                    ca = [(tiles[i][1][0][0]-tiles[i][1][2][0]) * zoom, (tiles[i][1][0][1]-tiles[i][1][2][1]) * zoom]
                    am = [xnow - (tiles[i][1][0][0] * zoom + move_x), ynow - (tiles[i][1][0][1] * zoom + move_y)]
                    bm = [xnow - (tiles[i][1][1][0] * zoom + move_x), ynow - (tiles[i][1][1][1] * zoom + move_y)]
                    cm = [xnow - (tiles[i][1][2][0] * zoom + move_x), ynow - (tiles[i][1][2][1] * zoom + move_y)]                   
                    if(ab[0]*am[1]-ab[1]*am[0] > 0 and bc[0]*bm[1]-bc[1]*bm[0] > 0 and ca[0]*cm[1]-ca[1]*cm[0] > 0):
                        #抓取圖片（xnow,ynow）位置的bgr值紀錄
                        bgr_total.append(img[ynow][xnow])
                    if(ab[0]*am[1]-ab[1]*am[0] < 0 and bc[0]*bm[1]-bc[1]*bm[0] < 0 and ca[0]*cm[1]-ca[1]*cm[0] < 0):
                        #抓取圖片（xnow,ynow）位置的bgr值紀錄
                        bgr_total.append(img[ynow][xnow])
                ynow = ynow + h_jump
            xnow = xnow + w_jump
            ynow = int(round(ymin, 0))
        # print("h_jump = ",h_jump)
        # print("w_jump = ",w_jump)
        # print("bgr_total.len = ",len(bgr_total))
        # print("bgr_total[0]",bgr_total[0])
        # print("bgr_total[1]",bgr_total[1])
        # print("bgr_total[2]",bgr_total[2])
        # print("bgr_total[3]",bgr_total[3])
        #累積bgr值找出最多的做為磚的顏色
        count_color = []
        most_color = 0
        color_find = False
        for m in range(len(bgr_total)):
            if(m == 0):
                count_color.append([bgr_total[m], 1])
                # print("count_color = ",count_color)
            else:
                color_find = False
                for n in range(len(count_color)):
                    # 附近像素顏色可能有極小差異
                    cc = ((bgr_total[m][0]-count_color[n][0][0])**2 + 
                          (bgr_total[m][1]-count_color[n][0][1])**2 +
                          (bgr_total[m][2]-count_color[n][0][2])**2 )**0.5
                    

                    if (cc < 15):
                        count_color[n][1] = count_color[n][1] + 1
                        if(count_color[n][1] > count_color[most_color][1] and most_color != n):
                            most_color = n
                        color_find = True
                        break
                # 與現有顏色不同
                if(color_find == False):
                    count_color.append([bgr_total[m], 1])
                
        # # # print(f"count_color : {len(count_color)}")
        # # # print(f"most_color : {most_color}")
        tiles_color.append(count_color[most_color][0])
    
    # print("tiles_color = ",tiles_color)
    # print("tiles_color.len = ",len(tiles_color))
    # print("tiles_color = ",tiles_color)
    print("tiles_color[1] = ",tiles_color[1]) # [237 245 248]
    print("type(tiles_color[1]) = ",type(tiles_color[1])) # [237 245 248]
    print("len(tiles) = ",len(tiles))
    # cv2.imshow("img",img)


    return tiles_color

def color_catch(trans, file_name, tiles): #抓取原圖中指定範圍內的所有像素點顏色
    
    img_name = file_name[:-4]
    png = [".jpeg", ".png", ".jpg", ".PNG", ".JPG"]
    find_file = False
    img = []
    for i in range(len(png)):
        if os.path.exists(img_name + png[i]):
            if os.path.isfile(img_name + png[i]):
                img = cv2.imread(img_name  + png[i])

                # 将图像转换为 LAB 空间
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                
                # image = cv2.imread(img_name + png[i])

                # slic = Slic(num_components=2000, compactness=150,subsample_stride=1,debug_mode=True)
                # assignment = slic.iterate(image) # Cluster Map
                # temp = slic.slic_model.clusters # The cluster information of superpixels.

                find_file = True
    if(find_file == False):
        print("no file")

    zoom = trans[0]
    move_x = trans[1]
    move_y = trans[2]
    tiles_color = []
    tiles_tmp = [] # tiles 轉換資料型態成 list
    # tiles_tmp.append(list(tiles))
    # print("tiles = ",tiles)
    # print("\n\n\n\n\n\n\n\n")

    tiles_tmp = copy.deepcopy(tiles)
    # print(tiles) # ('yellow', array([[2.5 , 0.9 ],[2.5 , 1.2 ],[2.75, 0.9 ],[2.5 , 0.9 ]]} 鋪磚狀況無顏色
    for i in range(len(tiles_tmp)):
        tiles_tmp[i] = list(tiles_tmp[i])
        # tiles_tmp[i].append("false") # init紀錄有沒有找過
        tiles_tmp[i][1][:,0] = tiles_tmp[i][1][:,0] * zoom + move_x # for why tiles & tiles-tmp 都一起被改了
        tiles_tmp[i][1][:,1] = tiles_tmp[i][1][:,1] * zoom + move_y
        # print(tiles_tmp[i][1]) # [[660.94005427 394.35479746] [660.94005427 426.79128225] [687.97045827 394.35479746] [660.94005427 394.35479746]]

        # # 計算資料集合的中心點
        # center = np.mean(tiles_tmp[i][1], axis=0)

        # # 將資料集合的每個點都放大兩倍
        # data_scaled = (tiles_tmp[i][1] - center) * 1.1 + center  

        # 建立一個二值化的區域 mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        pts = np.array(tiles_tmp[i][1]).astype(int)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1)

        masked_image = cv2.bitwise_and(img, img, mask=mask)

        # 使用 boolean indexing 篩選出區域內的像素值
        region_pixels = img[mask == 255]                      

        # 使用 k-means 聚合颜色
        kmeans = KMeans(n_clusters=10, random_state=0, init='k-means++').fit(region_pixels)


        # # 計算每個label的Silhouette Score
        # silhouette_scores = []
        # for i in range(5):
        #     silhouette_scores.append(silhouette_score(region_pixels, kmeans.labels_))
        #     print("silhouette_scores",silhouette_scores)
        #     # 如果所有樣本都屬於同一個聚類，則直接跳出迴圈
        #     if len(np.unique(kmeans.labels_)) == 1:
        #         break

        # 找出最大的Silhouette Score所對應的label
        # best_label = silhouette_scores.index(max(silhouette_scores))

        # # 计算每个点到其所属质心的距离平方和
        cluster_centers = kmeans.cluster_centers_
        distances = pairwise_distances_argmin_min(region_pixels, cluster_centers)
        sse = sum(distances[1])

        # 选择最小的SSE对应的标签
        best_label = kmeans.labels_[distances[0][np.argmin(distances[1])]]
        # best_label = get_best_label(region_pixels, 5, metric='silhouette')

        # 返回n個聚合過後的顏色值
        # 将图像转换回 BGR 空间
        bgr_pixel = cv2.cvtColor(np.uint8([cluster_centers]), cv2.COLOR_LAB2BGR)[0] # 下次看要不要改成不要轉成lab顏色空間
        print("bgr_pixel = ",bgr_pixel)
        print("bgr_pixel[best_label] = ",bgr_pixel[best_label])

        # # 找到每個元素的計數
        # unique_elements, counts = np.unique(bgr_pixel, axis=0, return_counts=True)

        # # 找到最大計數所對應的元素
        # most_common_element = unique_elements[np.argmax(counts)]

        tiles_color.append(bgr_pixel[best_label])

        # 建立一個黑色畫布
        canvas = np.zeros((300, 300, 3), dtype=np.uint8)

        # 使用 cv2.rectangle 函數畫正方形
        cv2.rectangle(canvas, (0, 0), (300, 300), (int(bgr_pixel[best_label][0]), int(bgr_pixel[best_label][1]), int(bgr_pixel[best_label][2])), thickness=-1)
        
        # cv2.imshow('Cropped Image', masked_image)
        # cv2.imshow("Square", canvas)
        # cv2.waitKey(0)
    
    # print("tiles_color = ",tiles_color)
    # print("tiles_color.len = ",len(tiles_color))
    # print("tiles_color = ",tiles_color)
    # print("tiles_color[1] = ",tiles_color[1]) # [237 245 248]
    # print("type(tiles_color[1]) = ",type(tiles_color[1])) # [237 245 248]
    # print("len(tiles) = ",len(tiles))
    # print("len(tiles_color) = ",len(tiles_color))


    return tiles_color

def get_best_label(X, max_k, metric='silhouette'):
    """
    根據指定的距離度量（silhouette_score或calinski_harabasz_score），
    從不同的聚類數目中選擇最佳的聚類數目。

    Parameters
    ----------
    X : array-like
        輸入的資料，形狀為 (n_samples, n_features)。
    max_k : int
        可能的最大聚類數目。
    metric : str
        聚類品質評估的度量方法。可選值有 'silhouette' 和 'calinski_harabasz'。

    Returns
    -------
    best_label : array-like
        最佳的聚類結果。

    """
    best_score = -np.inf
    best_label = None
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        labels = kmeans.labels_
        if len(set(labels)) == 1:
            break
        if metric == 'silhouette':
            score = silhouette_score(X, labels)
        elif metric == 'calinski_harabasz':
            score = calinski_harabasz_score(X, labels)
        if score > best_score:
            best_score = score
            best_label = labels
    return best_label

def find_lego_color(most_color):

    # #39種純色（b,g,r,編號）
    color_list = [  [244,244,244,1],
                    [141,185,204,5],
                    [90,128,187,18],
                    [0,0,180,21],
                    [168,90,30,23],
                    [10,200,250,24],
                    [29,19,5,26],
                    [43,133,0,28],
                    [65,171,88,37],
                    [28,8,145,38],
                    [200,150,115,102],
                    [35,121,214,106],
                    [24,202,165,119],
                    [118,31,144,124],
                    [154,129,112,135],
                    [98,125,137,138],
                    [90,50,25,140],
                    [26,69,0,141],
                    [124,142,112,151],
                    [18,0,114,154],
                    [0,172,252,191],
                    [9,49,95,192],
                    [150,150,150,194],
                    [100,100,100,199],
                    [247,195,157,212],
                    [157,53,211,221],
                    [205,15,255,222],
                    [108,236,255,226],
                    [145,26,68,268],
                    [149,201,255,283],
                    [0,33,53,308],
                    [85,125,170,312],
                    [195,155,70,321],
                    [226,195,104,322],
                    [234,242,211,323],
                    [185,110,160,324],
                    [222,164,205,325],
                    [154,249,226,326],
                    [79,132,139,330]]

    lego_color = 0
    dis = 10000


    # def ColourDistance(rgb_1, rgb_2):
    #     R_1,G_1,B_1 = rgb_1
    #     R_2,G_2,B_2 = rgb_2
    #     rmean = (R_1 +R_2 ) / 2
    #     R = R_1 - R_2
    #     G = G_1 -G_2
    #     B = B_1 - B_2
    #     return math.sqrt((2+rmean/256)*(R**2)+4*(G**2)+(2+(255-rmean)/256)*(B**2))


    
    for i in range(len(color_list)):
        '''
        dis_temp = ((most_color[0]-color_list[i][0])**2 + 
                    (most_color[1]-color_list[i][1])**2 +
                    (most_color[2]-color_list[i][2])**2 )**0.5
        '''

        # R_1 = most_color[2]
        # G_1 = most_color[1]
        # B_1 = most_color[0]
        # R_2 = color_list[i][2]
        # G_2 = color_list[i][1]
        # B_2 = color_list[i][0]
        # rmean = (R_1 +R_2 ) / 2
        # R = R_1 - R_2
        # G = G_1 -G_2
        # B = B_1 - B_2
        # dis_temp = math.sqrt((2+rmean/256)*(R**2)+4*(G**2)+(2+(255-rmean)/256)*(B**2))   

        dis_temp = (((most_color[0]-color_list[i][0])**2)*3 + 
                    ((most_color[1]-color_list[i][1])**2)*4 +
                    ((most_color[2]-color_list[i][2])**2)*2 )**0.5
        
        if (dis_temp < dis):
            dis = dis_temp
            lego_color = color_list[i][3]

    return lego_color
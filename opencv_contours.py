import numpy as np
import cv2
from maxrect import get_intersection, get_maximal_rectangle, rect2poly
from shapely.geometry import Polygon

# load image and shrink - it's massive
img = cv2.imread("/media/cglab/CEDCACB9DCAC9D69/TilinGNN-test/silhouette/bg3.png")
img = cv2.resize(img, None,fx=1, fy=1, interpolation = cv2.INTER_CUBIC)
print(type(img))
# get a blank canvas for drawing contour on and convert img to grayscale
canvas = np.zeros(img.shape, np.uint8)
img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# filter out small lines between counties
kernel = np.ones((5,5),np.float32)/25
img2gray = cv2.filter2D(img2gray,-1,kernel)

# threshold the image and extract contours
ret,thresh = cv2.threshold(img2gray,250,255,cv2.THRESH_BINARY_INV) ## 顏色二值化（黑色白色） 目的將圖片顏色落差太大的區分開成黑色白色
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# cv2.imshow('img',img)
# cv2.imshow('thresh',thresh)
# find the main island (biggest area)
cnt = contours[0]
max_area = cv2.contourArea(cnt)
for cont in contours: ### convex 演算法
    if cv2.contourArea(cont) > max_area:
        cnt = cont
        max_area = cv2.contourArea(cont)

# define main island contour approx. and hull
# perimeter = cv2.arcLength(cnt,True)
# epsilon = 0.01*cv2.arcLength(cnt,True)
# approx = cv2.approxPolyDP(cnt,epsilon,True)

# hull = cv2.convexHull(cnt)

def get_graph_bound(graph):

    x, y =  np.array(graph.exterior.coords.xy)
    x_min = np.min(x[:])
    x_max = np.max(x[:])
    y_min = np.min(y[:])
    y_max = np.max(y[:])

    return x_min, x_max, y_min, y_max

def init_(polynomial):

    center_point1 = polynomial.centroid
    center_point2 = (center_point1.x+1,center_point1.y)
    center_point3 = (center_point1.x+1,center_point1.y+1)
    center_point4 = (center_point1.x,center_point1.y+1)
    square = Polygon([center_point1,center_point2,center_point3,center_point4])
    
    return square

def add_(polynomial, deriction, off_x, off_y, times):
    x = []
    y = []

    x, y = polynomial.exterior.coords.xy
    # print("x, y =",x,y)
    if(deriction == 0): # 向右
        x[1] += (off_x * times)
        x[2] += (off_x * times)
    elif(deriction == 1): # 向下
        y[2] += (off_y * times)
        y[3] += (off_y * times)
    elif(deriction == 2): # 向左
        x[0] -= (off_x * times)
        x[4] -= (off_x * times)
        x[3] -= (off_x * times)
    elif(deriction == 3): # 向上
        y[0] -= (off_y * times)
        y[4] -= (off_y * times)
        y[1] -= (off_y * times)

    result = list(zip(x, y))

    return result

def maxsquare(polygons):
    square = init_(polygons)
    isIntersection = square.intersection(polygons).area
    x_min, x_max, y_min, y_max = get_graph_bound(polygons)
    unit_offset_x = (x_max - x_min) / 22
    unit_offset_y = (y_max - y_min) / 22

    print(unit_offset_x)
    print(unit_offset_y)

    print("isIntersection = ",isIntersection)

    for derict in range(4):
        print(derict)
        while isIntersection == square.area:
            temp = add_(square, derict, unit_offset_x, unit_offset_y, 1)
            temp = Polygon(temp) 
            temp_border = add_(temp, derict, unit_offset_x, unit_offset_y, 1)
            temp_border = Polygon(temp_border) 
            isIntersection_check = temp_border.intersection(polygons).area
            if(isIntersection_check < temp_border.area):
                break
            square = Polygon(temp)   
            isIntersection = square.intersection(polygons).area
            print("True")
        # temp = add_(square,x+4)
        # square = Polygon(temp) 
        # isIntersection = square.intersection(polygons).area
        

    if(isIntersection == square.area):
        print("Equal area.")

    return square


# cv2.isContourConvex(cnt)
cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 3)
# #cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 3)
# ## cv2.drawContours(canvas, hull, -1, (0, 0, 255), 3) # only displays a few points as well.
# cv2.polylines(canvas, [cnt.astype(int)], 1, (255, 0, 0), 2)
# cnt_list = np.array(cnt)[:, 0, :].tolist()
# coordinates3_poly = Polygon(cnt_list)
# square = maxsquare(coordinates3_poly)
# square_list = list(square.exterior.coords)
# square_array = np.array(square_list)
# square_array = np.expand_dims(square_array, axis=1)
# cv2.fillPoly(canvas,[square_array.astype(int)], (0, 255, 0))
# print(square_array)

# print("canvas = ",cnt_list)
# # print("canvas_type = ",np.shape(cnt_list))
# print("canvas_type = ",type(cnt_list))
# coordinates1_list = np.array(cnt_list)
# print("coordinates1_list_type = ",type(coordinates1_list))
# coordinates1_list = np.expand_dims(coordinates1_list, axis=1)

# cv2.fillPoly(canvas, [coordinates1_list.astype(int)], (0, 0, 255))

# _, coordinates = get_intersection([cnt_list])
# ll, ur = get_maximal_rectangle(coordinates)

# verts = np.array(rect2poly(ll, ur))
# verts = np.expand_dims(verts, axis=1)
# print(np.shape(verts))
# print(verts)
# cv2.fillPoly(canvas, [verts.astype(int)], (0, 255, 0))

cv2.imshow('canvas',canvas)
k = cv2.waitKey(0)
f = open("./silhouette/bg3.txt", "w")
for i in range(len(cnt)):
    if i == len(cnt) - 1:
        f.write(str(cnt[i][0][0]) + " " + str(cnt[i][0][1]) + '\n')
    else:
        f.write(str(cnt[i][0][0]) + " " + str(cnt[i][0][1])+",")
# for i in range(len(square_array)):
#     if i == len(square_array) - 1:
#         f.write(str(square_array[i][0][0]) + " " + str(square_array[i][0][1]))
#     else:
#         f.write(str(square_array[i][0][0]) + " " + str(square_array[i][0][1])+",")
f.close()

if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
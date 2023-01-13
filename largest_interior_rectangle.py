from maxrect import get_intersection, get_maximal_rectangle, rect2poly
import numpy as np
import cv2
from shapely.geometry import Polygon

# f = open('silhouette/shoes.txt', 'r')
# text = []
# for line in f:
#     text.append(line)
# print(text)
canvas = np.zeros((512,512,3), np.uint8)

# For a given convex polygon
coordinates3 = [ [0, 100], [200, 100], [100, 273.205080757]  ]
coordinates4 = [ [0, 150], [300, 150], [300, 450] ,[0, 450]  ]
coordinates5 = [ [0, 150],[200, 50], [400, 150], [300, 450] ,[100, 450]  ]
coordinates6 = [ [0, 150],[150, 50], [350, 50], [500, 150] ,[350, 450], [150, 450]  ]
coordinates7 = [ [0, 150],[150,50], [300, 150],[450,300], [300, 450] ,[150,500],[0, 450]  ]
# coordinates8 = [ [0, 150],[150, 50], [350, 50], [500, 150] ,[350, 450], [150, 450]  ]
coordinates9 = [ [0, 100], [66, 100], [100, 50], [132, 100], [200, 100], [166, 175], [200, 225], [132, 225],[100, 273.205080757]]
coordinates10 = [ [0, 100], [66, 100], [100, 50], [132, 100], [200, 100], [166, 175], [200, 225], [132, 225],[100, 273.205080757], [66, 225] ]
coordinates11 = [ [0, 100], [66, 100], [100, 50], [132, 100], [200, 100], [166, 175], [200, 225], [132, 225],[100, 273.205080757], [66, 225], [50, 225] ]
coordinates12 = [ [0, 100], [66, 100], [100, 50], [132, 100], [200, 100], [166, 175], [200, 225], [132, 225],[100, 273.205080757], [66, 225], [0, 225], [33, 175] ]
coordinates8 = [ [78, 32],[85, 32],[98, 33],[112, 32],[119, 34],[118, 42],[117, 59],[122, 74],[122, 89],[125, 94],[122, 105],[92, 107],[86, 105],[76, 102]
                ,[68, 105],[62, 108],[39, 109],[16, 102],[1, 97],[3, 90],[9, 85],[9, 78],[17, 76],[21, 76],[33, 75],[45, 71],[59, 65],[68, 56],[73, 46]  ]

# coordinates8_list = np.array(coordinates8)
# coordinates8_list = np.array(coordinates8)*3
# coordinates8 = coordinates8_list.tolist()
# coordinates8_list = np.expand_dims(coordinates8_list, axis=1)
# cv2.fillPoly(canvas,[coordinates8_list.astype(int)], (255, 0, 0))

coordinates3_list = np.array(coordinates8)
coordinates3_list = np.expand_dims(coordinates3_list, axis=1)
cv2.fillPoly(canvas,[coordinates3_list.astype(int)], (255, 0, 0))

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

def add_(polynomial, deriction, off_x, off_y):
    x = []
    y = []

    x, y = polynomial.exterior.coords.xy
    # print("x, y =",x,y)
    if(deriction == 0):
        x[1] += off_x
        x[2] += off_x
    elif(deriction == 1):
        y[2] += off_y
        y[3] += off_y
    elif(deriction == 2):
        x[0] -= off_x
        x[4] -= off_x
        x[3] -= off_x
    elif(deriction == 3):
        y[0] -= off_y
        y[4] -= off_y
        y[1] -= off_y # 修正 offset 的比例，可用最大/小x範圍 和 最大/小y範圍 除上我需要的樂高格子數

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

    for x in range(4):
        print(x)
        while isIntersection == square.area:
            temp = add_(square, x, unit_offset_x, unit_offset_y)
            temp_check = Polygon(temp) 
            isIntersection_check = temp_check.intersection(polygons).area
            if(isIntersection_check < temp_check.area):
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

coordinates3_poly = Polygon(coordinates8)
square = maxsquare(coordinates3_poly)
print("square = ",square)
square_list = list(square.exterior.coords)
print("square_list = ",square_list)
square_array = np.array(square_list)
square_array = np.expand_dims(square_array, axis=1)
cv2.fillPoly(canvas,[square_array.astype(int)], (0, 255, 0))

# # find the intersection of the polygons
# _, coordinates = get_intersection([coordinates8])

# # get the maximally inscribed rectangle
# ll, ur = get_maximal_rectangle(coordinates)

# # casting the rectangle to a GeoJSON-friendly closed polygon
# verts = np.array(rect2poly(ll, ur))
# verts = np.expand_dims(verts, axis=1)

# print(verts)

cv2.imshow('canvas',canvas)
k = cv2.waitKey(0)
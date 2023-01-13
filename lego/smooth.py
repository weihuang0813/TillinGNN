import numpy as np
import math
import sys
import similaritymeasures

lego_x = 0.25
lego_y = 0.3
max_distance = 0.5*(math.pow(lego_x,2) + math.pow(lego_y,2))**0.5 #離最近點的最大距離

# 變形用～按照ratio將原始輪廓轉換變形後輪廓
def nearest_of_points(origin_array, ratio):
    new_array = origin_array
    sign_x = 1
    sign_y = 1
    for i in range(len(origin_array)):
        sign_x = 1 if origin_array[i][0] > 0 else -1
        sign_y = 1 if origin_array[i][1] > 0 else -1
        if int(origin_array[i][0]/lego_x) == int(origin_array[i][0]/lego_x + 0.5*sign_x):
            nearest_x_ratio = int(origin_array[i][0]/lego_x)
        else:
            nearest_x_ratio = int(origin_array[i][0]/lego_x + 0.5*sign_x)
        if int(origin_array[i][1]/lego_y) == int(origin_array[i][1]/lego_y + 0.5*sign_y):
            nearest_y_ratio = int(origin_array[i][1]/lego_y)
        else:
            nearest_y_ratio = int(origin_array[i][1]/lego_y + 0.5*sign_y)

        distance = (math.pow(origin_array[i][0] - lego_x * nearest_x_ratio, 2) + \
                    math.pow(origin_array[i][1] - lego_y * nearest_y_ratio, 2)) ** 0.5
        if distance <= max_distance*((ratio)/100):
            new_array[i][0] = lego_x * nearest_x_ratio
            new_array[i][1] = lego_y * nearest_y_ratio
        else:
            continue
    #移除相同座標
    new_array = array_clear(new_array)
    
    return new_array

def array_clear(o_array):
    array_list = []
    if len(array_list) == 0:
        array_list.append(o_array[0])
    for i in range(len(o_array)):
        for j in range(len(array_list)):
            if (o_array[i] == array_list[j]).all():
                break
            if j == len(array_list) - 1 :
                array_list.append(o_array[i])
    array_list.append(array_list[0])
    
    return array_list

###############################################################################################################
# 相似度比較所使用的函數

def euc_dist(a,b):
    return math.sqrt( (a[0] - b[0])**2 + (a[1] - b[1])**2 )

def _c(ca, i, j, p, q):

    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = euc_dist(p[0],q[0])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i-1, 0, p, q), euc_dist(p[i],q[0]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j-1, p, q), euc_dist(p[0],q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(min(_c(ca, i-1, j, p, q),_c(ca, i-1, j-1, p, q),_c(ca, i, j-1, p, q)),euc_dist(p[i],q[j]))
    else:
        ca[i, j] = float('inf')

    return ca[i, j]

def frechet_distance(p,q):
    ca = np.ones((len(p),len(q)))
    ca = np.multiply(ca,-1)
    return _c(ca, len(p)-1, len(q)-1, p, q)


def similarity_compute(tiles_contour, shape_contour):
    sys.setrecursionlimit(5000)
    #tiles_contour = [[ 1.75, -1.8 ],[ 1.25, -2.1 ], ...[ 1.75, -1.8 ]]
    #shape_contour = [[-0.10163363, -2.4431244 ], ...[-0.10163363, -2.4431244 ]]
    
    similarity = 0     #最終輸出的相似度
    small_contour = [] #頂點數較少的輪廓
    large_contour = [] #頂點數較多的輪廓
    #'''
    if(len(tiles_contour) < len(shape_contour)):
        small_contour = tiles_contour
        large_contour = shape_contour
    else:
        small_contour = shape_contour
        large_contour = tiles_contour
    
    # 頂點少的去對應頂點多的

    # 先讓第一點找最近點
    length = 10000
    idx_t = 0
    idx = []    #所有small_contour座標對應的最近large_contour座標索引
    for i in range(len(small_contour)-1):
        length = 10000
        for j in range(len(large_contour)):
            length_t = math.sqrt( (small_contour[i][0] - large_contour[j][0])**2 + (small_contour[i][1] - large_contour[j][1])**2 )
            if(length_t < length):
                length = length_t
                idx_t = j
        idx.append(idx_t)
    idx.append(idx[0])
    '''
    # 第二點到第n點的最近點應出現在"前一點的最近點索引下deviation個點內找到"
    deviation = 0 
    if(len(small_contour) > 3):
        deviation = 3*int( len(large_contour)/len(small_contour) )
    else:
        deviation = len(large_contour)-1

    for i in range(len(small_contour)-2):  #扣除第一點和最後一點（因為第一點就是最後一點）
        length = 10000
        for j in range(deviation):
            length_t = math.sqrt( (small_contour[i+1][0] - large_contour[(j+idx[i])%len(large_contour)][0])**2 + 
                                  (small_contour[i+1][1] - large_contour[(j+idx[i])%len(large_contour)][1])**2 )
            if(length_t < length):
                length = length_t
                idx_t = (j+idx[i])%len(large_contour)
        idx.append(idx_t)
    '''

    # 各段輪廓相似度計算
    Fd = []
    for i in range(len(small_contour)-1):
        temp_contour1 = []
        temp_contour2 = []
        if(idx[i+1] == idx[i]+1):
            for j in range(2):
                temp_contour1.append(large_contour[j+idx[i]])
                temp_contour2.append(small_contour[i+j])
            Fd.append(frechet_distance(temp_contour1, temp_contour2))
        elif(idx[i+1] > idx[i]):
            insert_len = idx[i+1] - idx[i] - 1
            dis = [ small_contour[i+1][0] - small_contour[i][0], small_contour[i+1][1] - small_contour[i][1] ] #i到i+1的(x,y)距離
            temp_contour2.append(small_contour[i])
            for j in range(insert_len):
                temp_contour2.append( [ small_contour[i][0] + (dis[0]/(insert_len+1))*(j+1), 
                                        small_contour[i][1] + (dis[1]/(insert_len+1))*(j+1) ] )
            temp_contour2.append(small_contour[i+1])
            for k in range(insert_len+2):
                temp_contour1.append(large_contour[k+idx[i]])
            Fd.append(frechet_distance(temp_contour1, temp_contour2))
            
        elif(idx[i+1] < idx[i]):
            insert_len = len(large_contour) + idx[i+1] - idx[i] - 1
            dis = [ small_contour[i+1][0] - small_contour[i][0], small_contour[i+1][1] - small_contour[i][1] ] #i到i+1的(x,y)距離
            temp_contour2.append(small_contour[i])
            for j in range(insert_len):
                temp_contour2.append( [ small_contour[i][0] + (dis[0]/(insert_len+1))*(j+1), 
                                        small_contour[i][1] + (dis[1]/(insert_len+1))*(j+1) ] )
            temp_contour2.append(small_contour[i+1])
            for k in range(insert_len+2):
                temp_contour1.append(large_contour[(k+idx[i])%len(large_contour)])
            Fd.append(frechet_distance(temp_contour1, temp_contour2))
        else:
            # 兩個對應的large_contour索引相同
            temp_contour1.append(large_contour[idx[i]])
            temp_contour2.append(small_contour[i])
            temp_contour2.append(small_contour[i+1])
            Fd.append(frechet_distance(temp_contour1, temp_contour2))
    print(idx)
    print(Fd)
    similarity = sum(Fd)/len(Fd)

    '''
    # 將small_contour擴展為new_contour
    for i in range(len(small_contour)-1):
        
        insert_len = 0 
        new_contour.append(small_contour[i])
        if(idx[i+1] == idx[i]+1):
            continue
        elif(idx[i+1] > idx[i]):
            insert_len = idx[i+1] - idx[i] - 1
        elif(idx[i+1] < idx[i]):
            insert_len = len(large_contour) + idx[i+1] - idx[i] - 1
        else:
            # 兩個對應的large_contour索引相同
            continue
        dis = [ small_contour[i+1][0] - small_contour[i][0], small_contour[i+1][1] - small_contour[i][1] ] #i到i+1的(x,y)距離
        for j in range(insert_len):
            new_contour.append( [ small_contour[i][0] + (dis[0]/(insert_len+1))*(j+1), 
                                  small_contour[i][1] + (dis[1]/(insert_len+1))*(j+1) ] )
    new_contour.append(small_contour[len(small_contour)-1])
    print(idx)
    print(len(idx))
    print(large_contour)
    print(small_contour)
    print(new_contour)
    if(len(new_contour) != len(large_contour)):
        print("更新後兩輪廓還是不一樣長哭啊～！！！！！")
        print("////")
        print(len(small_contour))
        print(len(large_contour))
        print(len(new_contour))
        similarity = -1
    else:
        # 各段輪廓相似度計算
        temp_contour1 = []
        temp_contour2 = []
        Fd = []
        for i in range(len(small_contour)-1):
            for j in range(idx[i+1]-idx[i]):
                temp_contour1.append(large_contour[(j+idx[i])%len(large_contour)])
                temp_contour2.append(new_contour[(j+idx[i])%len(large_contour)])
            Fd.append(frechet_distance(temp_contour1, temp_contour2))
            print(frechet_distance(temp_contour1, temp_contour2))
        similarity = sum(Fd)/len(Fd)
    '''
    #'''
    return similarity

#純粹為了做消融實驗-無任何實質幫助
def similarity_compute2(tiles_contour, shape_contour):
    sys.setrecursionlimit(2000)
    #tiles_contour = [[ 1.75, -1.8 ],[ 1.25, -2.1 ], ...[ 1.75, -1.8 ]]
    #shape_contour = [[-0.10163363, -2.4431244 ], ...[-0.10163363, -2.4431244 ]]
    
    similarity = 0     #最終輸出的相似度
    small_contour = [] #頂點數較少的輪廓
    large_contour = [] #頂點數較多的輪廓
    new_contour = []
    #'''
    if(len(tiles_contour) < len(shape_contour)):
        small_contour = tiles_contour
        large_contour = shape_contour
    else:
        small_contour = shape_contour
        large_contour = tiles_contour
    
    # 頂點少的去對應頂點多的

    # 先讓第一點找最近點
    length = 10000
    idx_t = 0
    idx = []    #所有small_contour座標對應的最近large_contour座標索引
    for i in range(len(small_contour)-1):
        length = 10000
        for j in range(len(large_contour)):
            length_t = math.sqrt( (small_contour[i][0] - large_contour[j][0])**2 + (small_contour[i][1] - large_contour[j][1])**2 )
            if(length_t < length):
                length = length_t
                idx_t = j
        idx.append(idx_t)
    idx.append(idx[0])
    
    # 各段輪廓相似度計算
    # 將small_contour擴展為new_contour
    for i in range(len(small_contour)-1):
        
        insert_len = 0 
        new_contour.append(small_contour[i])
        if(idx[i+1] == idx[i]+1):
            continue
        elif(idx[i+1] > idx[i]):
            insert_len = idx[i+1] - idx[i] - 1
        elif(idx[i+1] < idx[i]):
            insert_len = len(large_contour) + idx[i+1] - idx[i] - 1
        else:
            # 兩個對應的large_contour索引相同
            continue
        dis = [ small_contour[i+1][0] - small_contour[i][0], small_contour[i+1][1] - small_contour[i][1] ] #i到i+1的(x,y)距離
        for j in range(insert_len):
            new_contour.append( [ small_contour[i][0] + (dis[0]/(insert_len+1))*(j+1), 
                                  small_contour[i][1] + (dis[1]/(insert_len+1))*(j+1) ] )
    new_contour.append(small_contour[len(small_contour)-1])

    similarity = frechet_distance(large_contour, new_contour)
    #'''
    return similarity

###############################################################################################################

## divide and conquer 

def center_move(center, lego_x, lego_y):
    sign_x = 1 if center[0] > 0 else -1
    sign_y = 1 if center[1] > 0 else -1
    if int(center[0]/lego_x) == int(center[0]/lego_x + 0.5*sign_x):
        x_temp = int(center[0]/lego_x) * lego_x
    else:
        x_temp = int(center[0]/lego_x + 0.5*sign_x) * lego_x
    if int(center[1]/lego_y) == int(center[1]/lego_y + 0.5*sign_y):
        y_temp = int(center[1]/lego_y) * lego_y
    else:
        y_temp = int(center[1]/lego_y + 0.5*sign_y) * lego_y

    x_move = x_temp - center[0]
    y_move = y_temp - center[1]

    return x_move, y_move
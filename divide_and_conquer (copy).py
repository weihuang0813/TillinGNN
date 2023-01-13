# 分割圖形測試
# 圖形的x軸向下,y軸向下

def Quadrant(x, y, center):
    if(x > center[0] and y < center[1]):
        return "1"
    elif(x < center[0] and y < center[1]):
        return "2"
    elif(x < center[0] and y > center[1]):
        return "3"
    elif(x > center[0] and y > center[1]):
        return "4"
    elif(x == center[0]):
        return "x"
    else:
        return "y"

def divide_and_conquer(contour_path):
    path = contour_path
    coordinate = []
    seg = []
    max_x = -10000
    min_x = 10000
    max_y = -10000
    min_y = 10000

    with open(path) as f:
        for line in f.readlines():
            s = line.replace(","," ").split(' ')
            if(len(s)%2 != 0):
                print("coordinate is wrong！!！")
            for i in range(int(len(s)/2)):
                coordinate.append([])
                coordinate[len(coordinate)-1].append(float(s[i*2]))
                coordinate[len(coordinate)-1].append(float(s[i*2+1]))
                if(float(s[i*2]) > max_x):
                    max_x = float(s[i*2])
                elif(float(s[i*2]) < min_x):
                    min_x = float(s[i*2])
                if(float(s[i*2+1]) > max_y):
                    max_y = float(s[i*2+1])
                elif(float(s[i*2+1]) < min_y):
                    min_y = float(s[i*2+1])

    center = ((max_x+min_x)/2 , (max_y+min_y)/2)
    print(f"divide center:{center}")
    temp = 0
    find = False
    round1 = True
    back = False
    now = " "
    end = (0,0)
    rem = (0,0)
    #找分歧點
    while(find == False):
        a = Quadrant(coordinate[temp][0], coordinate[temp][1], center)
        b = Quadrant(coordinate[temp+1][0], coordinate[temp+1][1], center)
        temp = temp + 1
        if(a != b):
            find = True
            break
    while(len(coordinate) != 0):
        #寫入第一點和最後一點
        if(b == 'x' or b == 'y'):
            if(round1 == False):
                seg[len(seg)-1].append([coordinate[temp][0], coordinate[temp][1]])
                seg[len(seg)-1].append([center[0], center[1]])
            seg.append([])
            seg[len(seg)-1].append([center[0], center[1]])
            seg[len(seg)-1].append([coordinate[temp][0], coordinate[temp][1]])
            del coordinate[temp]
        else:
            if(round1 == False):
                if((a=="2" and b=="1") or (a=="4" and b=="3") or (a=="1" and b=="2") or (a=="3" and b=="4")):
                    seg[len(seg)-1].append([center[0], (end[1]+coordinate[temp][1])/2])
                else:
                    seg[len(seg)-1].append([(end[0]+coordinate[temp][0])/2, center[1]])
                seg[len(seg)-1].append([center[0], center[1]])
                seg.append([])
                seg[len(seg)-1].append([center[0], center[1]])
                if((a=="2" and b=="1") or (a=="4" and b=="3") or (a=="1" and b=="2") or (a=="3" and b=="4")):
                    seg[len(seg)-1].append([center[0], (end[1]+coordinate[temp][1])/2])
                else:
                    seg[len(seg)-1].append([(end[0]+coordinate[temp][0])/2, center[1]])
            else:
                seg.append([])
                seg[len(seg)-1].append([center[0], center[1]])
                if((a=="2" and b=="1") or (a=="4" and b=="3") or (a=="1" and b=="2") or (a=="3" and b=="4")):
                    seg[len(seg)-1].append([center[0], (coordinate[temp-1][1]+coordinate[temp][1])/2])
                else:
                    seg[len(seg)-1].append([(coordinate[temp-1][0]+coordinate[temp][0])/2, center[1]])

        if(round1 == True):
            rem = seg[0][1]
            round1 = False
        
        if(b == "x" or b == "y"):
            if(coordinate[temp] == coordinate[0] and back == False):
                del coordinate[temp]
                temp = 0
                back = True
            now = Quadrant(coordinate[temp+1][0], coordinate[temp+1][1], center)
        else:
            now = b
        #寫入中間所有點
        while(find == True):     
            if(temp == len(coordinate) and len(coordinate) != 0):
                temp = 0
                back = True
            elif(coordinate[temp] == coordinate[0] and back == False):
                del coordinate[temp]
                temp = 0
                back = True
                
            c = Quadrant(coordinate[temp][0], coordinate[temp][1], center)
            if(c == now):
                end = (coordinate[temp][0], coordinate[temp][1])
                seg[len(seg)-1].append([coordinate[temp][0], coordinate[temp][1]])
                del coordinate[temp]
            else:
                a = b
                b = c
                break
            if(len(coordinate) == 0):
                a = now
                b = Quadrant(rem[0], rem[1], center)
                if(b == "x" or b == "y"):
                    seg[len(seg)-1].append([rem[0], rem[1]])
                else:
                    if((a=="2" and b=="1") or (a=="4" and b=="3") or (a=="1" and b=="2") or (a=="3" and b=="4")):
                        seg[len(seg)-1].append([center[0], (end[1]+rem[1])/2])
                    else:
                        seg[len(seg)-1].append([(end[0]+rem[0])/2, center[1]])
                seg[len(seg)-1].append([center[0], center[1]])
                break
    for j in range(4):
        #print("######################")
        #print(seg[j])
        Q = Quadrant(seg[j][int(len(seg[j])/2)][0], seg[j][int(len(seg[j])/2)][1], center)
        ff = open("Divide_" + str(int(Q)) +".txt",'w+')
        for i in range(len(seg[j])):
            ff.write(str(seg[j][i][0]) + " " + str(seg[j][i][1]))
            if(i == len(seg[j])-1):
                break
            ff.write(",")
    ff.close
    return 0


# main
p = 'silhouette/circle.txt'
divide_and_conquer(p)
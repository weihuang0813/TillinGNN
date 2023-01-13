# random merge

'''
0 (可讀入input名稱)
1  4 -10.0 12.0 0 1 0 0 0 1 0 0 0 1 3005.DAT
1  4 10.0 12.0 0 1 0 0 0 1 0 0 0 1 3005.DAT
1  4 -50.0 60.0 0 0 0 -1 0 1 0 1 0 0 4286.DAT...
'''

import random
origin_brick = [] #（x座標,y座標,brick name,color）(float,float,string,float)(3,4,15,2)
slope_list =[]    #記錄原檔案不會改變的斜磚資訊
sort_brick = []   #以y座標分類再以x座標排序所有基本磚
color_brick = []
result_brick = [] #紀錄最終要輸出的樂高磚資訊（x座標,y座標,brick name,磚的長度）(float,float,string,int)
first_time = True
ldr_x = 20
ldr_y = 24
ymin = 1000
ymax = -1000

# 判斷所選的磚merge後與下層的連接性
# 輸入：選到的磚的索引, 第i層的磚的x座標, 第i-1層的result_brick, can_merge_idx, 磚的顏色 (int,[],[],[],float)
# 輸出：所有候選的磚(起始idx,brick name,長度)([[int,string,int],...])
def brick_connection(idx, brick_x, previous_layer, con_idx, brick_color):

    score = 0        #目前優先度(即連結下層的磚的數量)
    temp_score = 0   #預計優先度
    candidate = []
    pre_layer_x = [] #紀錄所有間距（每個間距前後分別為x座標的左右極限，中間皆為相鄰處）

    #將第i-1層的result_brick的左右間距和相鄰處列出，若不相鄰則有許多組間距
    for a in range(int(len(previous_layer)/4)):
        left_x  = previous_layer[0+4*a] - ldr_x*previous_layer[3+4*a]/2
        right_x = previous_layer[0+4*a] + ldr_x*previous_layer[3+4*a]/2
        pre_layer_x.append([left_x,right_x])

    if(len(con_idx) >= 1):
        temp_score = 0
        left_x = brick_x[idx] - ldr_x/2
        right_x = brick_x[idx] + ldr_x/2
        for a in range(len(pre_layer_x)):
            if(pre_layer_x[a][0] <= left_x and right_x <= pre_layer_x[a][1]):
                temp_score = 1
                break
        if(temp_score > score):
            candidate = []
            score = temp_score
            candidate.append([idx,'3005.DAT',1])
        elif(temp_score == score):
            candidate.append([idx,'3005.DAT',1])
    if(len(con_idx) >= 2):
        for a in range(2):
            temp_score = 0
            if(idx-a >= con_idx[0] and idx-a+1 <= con_idx[len(con_idx)-1]):
                left_x = brick_x[idx-a] - ldr_x/2
                right_x = brick_x[idx-a+1] + ldr_x/2
                for b in range(len(pre_layer_x)):
                    if(pre_layer_x[b][0] <= left_x and right_x <= pre_layer_x[b][1]):
                        temp_score = 1
                        break
                    elif(left_x < pre_layer_x[b][0] < right_x or left_x < pre_layer_x[b][1] < right_x):
                        temp_score = temp_score + 1
                if(temp_score > score):
                    candidate = []
                    score = temp_score
                    candidate.append([idx-a,'3004.DAT',2])
                elif(temp_score == score):
                    candidate.append([idx-a,'3004.DAT',2])
            
    if(len(con_idx) >= 3):
        for a in range(3):
            temp_score = 0
            if(idx-a >= con_idx[0] and idx-a+2 <= con_idx[len(con_idx)-1]):
                left_x = brick_x[idx-a] - ldr_x/2
                right_x = brick_x[idx-a+2] + ldr_x/2
                for b in range(len(pre_layer_x)):
                    if(pre_layer_x[b][0] <= left_x and right_x <= pre_layer_x[b][1]):
                        temp_score = 1
                        break
                    elif(left_x < pre_layer_x[b][0] < right_x or left_x < pre_layer_x[b][1] < right_x):
                        temp_score = temp_score + 1
                if(temp_score > score):
                    candidate = []
                    score = temp_score
                    candidate.append([idx-a,'3622.DAT',3])
                elif(temp_score == score):
                    candidate.append([idx-a,'3622.DAT',3])
    if(len(con_idx) >= 4):
        for a in range(4):
            temp_score = 0
            if(idx-a >= con_idx[0] and idx-a+3 <= con_idx[len(con_idx)-1]):
                left_x = brick_x[idx-a] - ldr_x/2
                right_x = brick_x[idx-a+3] + ldr_x/2
                for b in range(len(pre_layer_x)):
                    if(pre_layer_x[b][0] <= left_x and right_x <= pre_layer_x[b][1]):
                        temp_score = 1
                        break
                    elif(left_x < pre_layer_x[b][0] < right_x or left_x < pre_layer_x[b][1] < right_x):
                        temp_score = temp_score + 1
                if(temp_score > score):
                    candidate = []
                    score = temp_score
                    candidate.append([idx-a,'3010.DAT',4])
                elif(temp_score == score):
                    candidate.append([idx-a,'3010.DAT',4])
    if(len(con_idx) >= 6):
        for a in range(6):
            temp_score = 0
            if(idx-a >= con_idx[0] and idx-a+5 <= con_idx[len(con_idx)-1]):
                left_x = brick_x[idx-a] - ldr_x/2
                right_x = brick_x[idx-a+5] + ldr_x/2
                for b in range(len(pre_layer_x)):
                    if(pre_layer_x[b][0] <= left_x and right_x <= pre_layer_x[b][1]):
                        temp_score = 1
                        break
                    elif(left_x < pre_layer_x[b][0] < right_x or left_x < pre_layer_x[b][1] < right_x):
                        temp_score = temp_score + 1
                if(temp_score > score):
                    candidate = []
                    score = temp_score
                    candidate.append([idx-a,'3009.DAT',6])
                elif(temp_score == score):
                    candidate.append([idx-a,'3009.DAT',6])
    if(len(con_idx) >= 8):
        for a in range(8):
            temp_score = 0
            if(idx-a >= con_idx[0] and idx-a+7 <= con_idx[len(con_idx)-1]):
                left_x = brick_x[idx-a] - ldr_x/2
                right_x = brick_x[idx-a+7] + ldr_x/2
                for b in range(len(pre_layer_x)):
                    if(pre_layer_x[b][0] <= left_x and right_x <= pre_layer_x[b][1]):
                        temp_score = 1
                        break
                    elif(left_x < pre_layer_x[b][0] < right_x or left_x < pre_layer_x[b][1] < right_x):
                        temp_score = temp_score + 1
                if(temp_score > score):
                    candidate = []
                    score = temp_score
                    candidate.append([idx-a,'3008.DAT',8])
                elif(temp_score == score):
                    candidate.append([idx-a,'3008.DAT',8])

    return candidate


# main function 

path = 'unity_use/node_pos_test.txt'
with open(path) as f:
    for line in f.readlines():
        if(first_time == True):
            first_time = False
            continue
        s = line.split(' ')
        if(s[15] != '3005.DAT\n'):
            slope_list.append(line)
        else:
            origin_brick.append([])
            origin_brick[len(origin_brick)-1].append(float(s[3]))
            origin_brick[len(origin_brick)-1].append(float(s[4]))
            origin_brick[len(origin_brick)-1].append(s[15].replace('.DAT\n', ''))
            origin_brick[len(origin_brick)-1].append(float(s[2]))
            if(float(s[4]) < ymin):
                ymin = float(s[4])
            elif(float(s[4]) > ymax):
                ymax = float(s[4])

#檢驗ymax和ymin的差距是否為24的倍數
check = (ymax-ymin) / ldr_y
if( check != int(check) ):
    print("樂高磚位置有問題？？？")

#將origin_brick中的所有基本磚以y座標分類再以x座標做排序
for i in range(int(check) + 1 ):
    sort_brick.append([])
    color_brick.append([])
for i in range(len(origin_brick)):
    if(origin_brick[i][2] != '3005'):
        continue
    pos = int((ymax - origin_brick[i][1])/ldr_y)
    if(origin_brick[i][0] > 0):
        sort_brick[pos].append(origin_brick[i][0])
        color_brick[pos].append(origin_brick[i][3])
    else:
        sort_brick[pos].insert(0,origin_brick[i][0])
        color_brick[pos].insert(0,origin_brick[i][3])
'''
#檢查是否有按照x座標排序
for i in range(len(sort_brick)):
    for j in range(len(sort_brick[i])-1):
        if(sort_brick[i][j] > sort_brick[i][j+1]):
            temp = sort_brick[i][j]
            sort_brick[i][j] = sort_brick[i][j+1]
            sort_brick[i][j+1] = temp
            print("排序有錯 已修正")
'''

#主要random merge
for i in range(len(sort_brick)):
    result_brick.append([])
    while(len(sort_brick[i]) != 0): 
        can_merge_idx = []
        choice = random.randrange(0,len(sort_brick[i])) #每層所隨機選的磚(只會選擇未合併的,已合併或無法合併的將逐漸被移除)
        choice2 = 0
        can_merge_idx.append(choice)
        for j in range(len(sort_brick[i])-choice-1):
            if(sort_brick[i][choice+1+j] - sort_brick[i][choice+j] != ldr_x or color_brick[i][choice+1+j] != color_brick[i][choice+j]):
                break
            else:
                can_merge_idx.append(choice+1+j)
        for k in range(choice):
            if(sort_brick[i][choice-k] - sort_brick[i][choice-1-k] != ldr_x or color_brick[i][choice-k] != color_brick[i][choice-1-k]):
                break
            else:
                can_merge_idx.insert(0,choice-1-k)
        #其他層：隨機選磚，連結底下越多磚越好(優先度越高)
        if(i != 0):
            #紀錄所有優先度最高的磚,用於刪除sort_brick[i]的索引(起始idx,brick name,長度)
            best_brick = brick_connection(choice, sort_brick[i], result_brick[i-1], can_merge_idx, color_brick[i])
            #最後隨機選擇best_brick其中一個,再輸出至result_brick,並移除sort_brick[i]的索引
            if(len(best_brick) != 1):
                choice2 = random.randrange(0,len(best_brick))
                result_brick[i].append(sort_brick[i][best_brick[choice2][0]] + (best_brick[choice2][2]-1)*ldr_x/2)
                result_brick[i].append(ymax - ldr_y*i)
                result_brick[i].append(best_brick[choice2][1])
                result_brick[i].append(color_brick[i][choice])
                del sort_brick[i][best_brick[choice2][0]:best_brick[choice2][0]+best_brick[choice2][2]]
                del color_brick[i][best_brick[choice2][0]:best_brick[choice2][0]+best_brick[choice2][2]]
            else:
                result_brick[i].append(sort_brick[i][best_brick[0][0]] + (best_brick[0][2]-1)*ldr_x/2)
                result_brick[i].append(ymax - ldr_y*i)
                result_brick[i].append(best_brick[0][1])
                result_brick[i].append(color_brick[i][choice])
                del sort_brick[i][best_brick[0][0]:best_brick[0][0]+best_brick[0][2]]
                del color_brick[i][best_brick[0][0]:best_brick[0][0]+best_brick[0][2]]
        #最底層：隨機選磚，磚越長越好
        else:
            if(len(can_merge_idx) >= 8):
                if(len(can_merge_idx) != 8):
                    while(not(choice2 <= choice <= choice2+7)):
                        choice2 = random.randrange(0,len(can_merge_idx)-8+1)
                result_brick[i].append((sort_brick[i][can_merge_idx[0+choice2]] + sort_brick[i][can_merge_idx[7+choice2]]) / 2)
                result_brick[i].append(ymax - ldr_y*i)
                result_brick[i].append('3008.DAT')
                result_brick[i].append(color_brick[i][choice])
                del sort_brick[i][can_merge_idx[0]+0+choice2:can_merge_idx[0]+8+choice2]
                del color_brick[i][can_merge_idx[0]+0+choice2:can_merge_idx[0]+8+choice2]
            elif(len(can_merge_idx) >= 6):
                if(len(can_merge_idx) != 6):
                    while(not(choice2 <= choice <= choice2+5)):
                        choice2 = random.randrange(0,2)
                result_brick[i].append((sort_brick[i][can_merge_idx[0+choice2]] + sort_brick[i][can_merge_idx[5+choice2]]) / 2)
                result_brick[i].append(ymax - ldr_y*i)
                result_brick[i].append('3009.DAT')
                result_brick[i].append(color_brick[i][choice])
                del sort_brick[i][can_merge_idx[0]+0+choice2:can_merge_idx[0]+6+choice2]
                del color_brick[i][can_merge_idx[0]+0+choice2:can_merge_idx[0]+6+choice2]
            elif(len(can_merge_idx) >= 4):
                if(len(can_merge_idx) != 4):
                    while(not(choice2 <= choice <= choice2+3)):
                        choice2 = random.randrange(0,2)
                result_brick[i].append((sort_brick[i][can_merge_idx[0+choice2]] + sort_brick[i][can_merge_idx[3+choice2]]) / 2)
                result_brick[i].append(ymax - ldr_y*i)
                result_brick[i].append('3010.DAT')
                result_brick[i].append(color_brick[i][choice])
                del sort_brick[i][can_merge_idx[0]+choice2:can_merge_idx[0]+4+choice2]
                del color_brick[i][can_merge_idx[0]+choice2:can_merge_idx[0]+4+choice2]
            elif(len(can_merge_idx) == 3):
                result_brick[i].append(sort_brick[i][can_merge_idx[1]])
                result_brick[i].append(ymax - ldr_y*i)
                result_brick[i].append('3622.DAT')
                result_brick[i].append(color_brick[i][choice])
                del sort_brick[i][can_merge_idx[0]:can_merge_idx[0]+len(can_merge_idx)]
                del color_brick[i][can_merge_idx[0]:can_merge_idx[0]+len(can_merge_idx)]
            elif(len(can_merge_idx) == 2): 
                result_brick[i].append( (sort_brick[i][can_merge_idx[0]] + sort_brick[i][can_merge_idx[1]]) / 2)
                result_brick[i].append(ymax - ldr_y*i)
                result_brick[i].append('3004.DAT')
                result_brick[i].append(color_brick[i][choice])
                del sort_brick[i][can_merge_idx[0]:can_merge_idx[0]+len(can_merge_idx)]
                del color_brick[i][can_merge_idx[0]:can_merge_idx[0]+len(can_merge_idx)]
            elif(len(can_merge_idx) == 1):
                result_brick[i].append(sort_brick[i][can_merge_idx[0]])
                result_brick[i].append(ymax - ldr_y*i)
                result_brick[i].append('3005.DAT')
                result_brick[i].append(color_brick[i][choice])
                del sort_brick[i][can_merge_idx[0]]
                del color_brick[i][can_merge_idx[0]]

#print(result_brick)

#寫檔
#ff = open("new_rectangle.txt",'w+')
ff = open("unity_use/new_node_pos.txt",'w+')
ff.write("0 (可讀入input名稱)" + "\n")
for i in range(len(slope_list)):
    ff.write(str(slope_list[i]))
for i in range(len(result_brick)):
    for j in range(int(len(result_brick[i])/4)):
        ff.write("1  "+ str(int(result_brick[i][3+4*j])) + " " + str(result_brick[i][0+4*j]) + " " + str(result_brick[i][1+4*j]) + " 0 " + "1 0 0 0 1 0 0 0 1 " + str(result_brick[i][2+4*j]))
        # 顏色 x軸 y軸 方塊代碼
        ff.write("\n")
ff.close

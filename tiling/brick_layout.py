import os
import numpy as np
from tiling.tile_graph import TileGraph
from shapely.ops import unary_union
import shapely
from shapely.geometry import Polygon
from collections import defaultdict
import util.data_util
import copy
from util import fabrication
from util.algo_util import interp
import random
import networkx as nx
import itertools
from util.tiling_util import polygon_align_length
import matplotlib.pyplot as plt
import math
from lego.color import color_catch, find_lego_color
import time
import similaritymeasures
import matplotlib.pyplot as plt
from lego.color import calculation_of_transform
from lego.smooth import frechet_distance, similarity_compute, similarity_compute2
#from lego.smooth import similarity_compute

# episolon for area err
EPS = 1e-5
BUFFER_TILE_EPS = EPS * 1e-5
SIMPLIFIED_TILE_EPS = BUFFER_TILE_EPS * 1e3

class BrickLayout():
    def __init__(self, complete_graph: TileGraph, node_feature, collide_edge_index,
                 collide_edge_features, align_edge_index, align_edge_features, re_index,
                 target_polygon=None):
        self.complete_graph = complete_graph
        self.node_feature = node_feature
        self.collide_edge_index = collide_edge_index
        self.collide_edge_features = collide_edge_features
        self.align_edge_index = align_edge_index
        self.align_edge_features = align_edge_features

        ## assertion for brick_layout
        align_edge_index_list = align_edge_index.T.tolist()
        collide_edge_index_list = collide_edge_index.T.tolist()

        ## assertion
        # for f in align_edge_index_list:
        #     assert [f[1], f[0]] in align_edge_index_list
        # for f in collide_edge_index_list:
        #     assert [f[1], f[0]] in collide_edge_index_list

        # mapping from index of complete graph to index of super graph
        self.re_index = re_index
        # mapping from index of super graph to index of complete graph
        self.inverse_index = defaultdict(int)
        for k, v in self.re_index.items():
            self.inverse_index[v] = k

        self.predict = np.zeros(len(self.node_feature))
        self.predict_probs = []
        self.predict_order = []
        self.target_polygon = target_polygon

        ### save super poly
        self.super_contour_poly = None

    def __deepcopy__(self, memo):
        new_inst = type(self).__new__(self.__class__)  # skips calling __init__
        new_inst.complete_graph = self.complete_graph
        new_inst.node_feature = self.node_feature
        new_inst.collide_edge_index = self.collide_edge_index
        new_inst.collide_edge_features = self.collide_edge_features
        new_inst.align_edge_index = self.align_edge_index
        new_inst.align_edge_features = self.align_edge_features
        new_inst.re_index = self.re_index
        new_inst.inverse_index = self.inverse_index
        new_inst.predict = copy.deepcopy(self.predict)
        new_inst.predict_probs = copy.deepcopy(self.predict_probs)
        new_inst.super_contour_poly = self.super_contour_poly
        new_inst.predict_order = self.predict_order
        new_inst.target_polygon = self.target_polygon

        return new_inst

    def is_solved(self):
        return len(self.predict) != 0

    ############ ALL PLOTTING FUNCTIONS #################
    def show_candidate_tiles(self, plotter, debugger, file_name, style ="blue_trans"):
        tiles = self.complete_graph.tiles
        selected_indices = [k for k in self.re_index.keys()]
        selected_tiles = [tiles[s] for s in selected_indices]
        plotter.draw_contours(debugger.file_path(file_name),
                              [tile.get_plot_attribute(style) for tile in selected_tiles])

    def show_predict(self, plotter, debugger, file_name, do_show_super_contour = True, do_show_tiling_region = True):
        outter = []
        inner = []        
        tiles = self.predict

        # show input polygon
        tiling_region_exteriors, tiling_region_interiors = BrickLayout.get_polygon_plot_attr(self.target_polygon) \
                                                               if do_show_tiling_region else ([],[])

        # show cropped region
        super_contour_poly = self.get_super_contour_poly()
        super_contour_exteriors, super_contour_interiors = BrickLayout.get_polygon_plot_attr(super_contour_poly, style='lightblue') \
                                                               if do_show_super_contour else ([], [])
        # show selected tiles
        selected_tiles = [self.complete_graph.tiles[self.inverse_index[i]].get_plot_attribute("yellow") for i in
                               range(len(tiles)) if tiles[i] == 1]

        plotter.draw_contours(debugger.file_path(file_name),
                              tiling_region_exteriors + tiling_region_interiors + super_contour_exteriors + super_contour_interiors + selected_tiles)
        
        def ploy_iou(poly1:Polygon,poly2:Polygon):
            intersection_area = poly1.intersection(poly2).area
            union_area = poly1.union(poly2).area
            return intersection_area / union_area
        
        if(tiling_region_interiors != [] and super_contour_interiors != []):
            for i in range(len(tiling_region_interiors[0])):                     
                outter.append((tiling_region_interiors[0][1] * 100).astype(int).tolist())

            for i in range(len(super_contour_interiors[0])):
                try:
                    super_contour_interiors[i][1]
                except IndexError:             #  super_contour_interiors 索引值裡面有sample完的痕跡  
                    break                      #  導致會有index超出的狀況  
                inner.append((super_contour_interiors[i][1] * 100).astype(int).tolist())
        
        west_exterior = (tiling_region_exteriors[0][1] * 100).astype(int).tolist()
        west_poly = Polygon(shell=west_exterior, holes=outter)

        west_exterior1 = (super_contour_exteriors[0][1] * 100).astype(int).tolist()
        west_poly1 = Polygon(shell=west_exterior1, holes=inner)

        print('IOU Value = ', ploy_iou(west_poly, west_poly1))
        
    def show_predict_for_ldr(self, plotter, debugger, ori_name, offset, file_name, do_show_super_contour = True, do_show_tiling_region = True):
        ##當input不同時要對此塊程式做修改
        tiles = self.predict
        feature = self.node_feature

        selected_feature = [np.ndarray.tolist(feature[i]) for i in range(len(feature)) if tiles[i] == 1]

        ## 輸入圖形的輪廓
        tiling_region_exteriors, tiling_region_interiors = BrickLayout.get_polygon_plot_attr(self.target_polygon) \
                                                               if do_show_tiling_region else ([],[])
        '''
        tiling_region_exteriors = [('light_gray', array([[-0.10163363, -2.4431244 ], ...[-0.10163363, -2.4431244 ]]))]
        tiling_region_interiors = [('white', array([[1.77691209, 0.23278965], ...[1.77691209, 0.23278965]])), ...
                                   ('white', array([[-1.58825255,  0.30036323], ...[-1.58825255,  0.30036323]]))]

        tiling_region_exteriors[0][1][0] = [-0.10163363, -2.4431244 ]
        '''

        # 從這邊加入內輪廓最靠近樂高磚的正方形頂點，在將所有正方形內填滿集完成結案


        plotter.draw_contours(debugger.file_path("input_" + file_name),
                            tiling_region_exteriors + tiling_region_interiors)
        # show cropped region
        super_contour_poly = self.get_super_contour_poly()
        super_contour_exteriors, super_contour_interiors = BrickLayout.get_polygon_plot_attr(super_contour_poly, style='lightblue') \
                                                               if do_show_super_contour else ([], [])
        # show selected tiles
        selected_tiles = [self.complete_graph.tiles[self.inverse_index[i]].get_plot_attribute("yellow") for i in
                               range(len(tiles)) if tiles[i] == 1]
        '''
        selected_tiles = [('yellow', array([[0.5 , 0.3 ],[0.5 , 0.6 ],[0.75, 0.6 ],[0.75, 0.3 ],[0.5 , 0.3 ]])), ...]
        '''

        if(tiling_region_interiors != []): # 用小矩形平鋪大矩形內

            print("tiling_region_interiors = ",tiling_region_interiors)
            tiling_region_interiors_border = tiling_region_interiors[0][1]
            ppx, ppy = zip(*tiling_region_interiors_border)
            
            square_maxx = math.ceil(4 * max(ppx)) * 0.25
            square_maxy = math.ceil((10/3) * max(ppy)) * 0.3
            square_minx = math.ceil(4 * abs(min(ppx))) * -0.25
            square_miny = math.ceil((10/3) * abs(min(ppy))) * -0.3

            # print(square_maxx,square_maxy)
            # print(square_minx,square_miny)

            for i in np.arange(square_minx,square_maxx,0.25):
                for j in np.arange(square_miny,square_maxy,0.3):
                    
                    selected_tiles.append(['yellow',np.array([[i,j],[i,j+0.3],[i+0.25,j+0.3]
                                                ,[i+0.25,j],[i,j]])])
                    selected_feature.append([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.9999999999999989])
                    # print(i,j)
        
        # print(selected_tiles)
        print(len(selected_tiles))
        # print(type(selected_tiles))
        # print(type(selected_tiles[0][1]))
        # 顏色
        # '''
        color_num = []       #樂高的顏色編號
        temp_color_find = [] #暫時計憶曾經搜尋過的顏色
        trans = calculation_of_transform(ori_name, tiling_region_exteriors[0][1])
        print("trans = ",trans)
        lego_color = color_catch(trans, ori_name, selected_tiles)
        print("====================================lego_color=======================================")
        # print(lego_color)
        print(len(lego_color))
        if(len(lego_color) == len(selected_tiles)):
            for i in range(len(lego_color)):
                find = False
                if(len(temp_color_find) != 0):
                    for j in range(len(temp_color_find)):
                        if(temp_color_find[j][0].all == lego_color[i].all):
                            color_num.append(temp_color_find[j][1])
                            find = True
                            break
                if(find == False):
                    cc = find_lego_color(lego_color[i])
                    color_num.append(cc)
                    temp_color_find.append([])
                    temp_color_find[len(temp_color_find)-1].append(lego_color[i])
                    temp_color_find[len(temp_color_find)-1].append(cc)
        else:
            print("color_catch wrong!!!")
        # print(f"color_num : {color_num}")
        # '''

        # 樂高磚的輪廓
        poly = []
        square = [] # 處理inner square 填磚
        for i in range(len(selected_tiles)):
            round_tiles = []
            for j in range(len(selected_tiles[i][1])):
                square.append(list(selected_tiles[i][1][j]))
                r = [round(selected_tiles[i][1][j][0],2), round(selected_tiles[i][1][j][1],2)]
                round_tiles.append(r)
            poly.append(Polygon(round_tiles))
        total_polygon = unary_union(poly)

        square.sort()
        ppx, ppy = zip(*square)
        # tiles_exteriors_contour_list（外輪廓）, tiles_interiors_list（內輪廓）用於比較相似度
        tiles_exteriors_contour_list, tiles_interiors_contour_list = BrickLayout.get_polygon_plot_attr(total_polygon, show_line = True)
        '''
        tiles_exteriors_contour_list = [('light_gray', array([[ 1.75, -1.8 ],[ 1.25, -2.1 ], ...[ 1.75, -1.8 ]]))]
        tiles_interiors_contour_list = [('white', array([[1.5 , 0.  ],[2.  , 0.3 ], ...[1.5 , 0.  ]])), ...
                                        ('white', array([[-1.5 , 0.9 ],[-1.75,  0.3 ], ...[-1.5 ,  0.9 ]]))]

        tiles_exteriors_contour_list[0][1][0] = [ 1.75, -1.8 ]
        '''

        ## 樂高磚的輪廓繪製
        plotter.draw_contours(debugger.file_path("tiles_" + file_name),
                            tiles_exteriors_contour_list + tiles_interiors_contour_list + selected_tiles)
        
        ## 輪廓相似度比較
        #print(f"樂高磚輪廓: {tiles_exteriors_contour_list[0][1]}")
        #print(f"原輪廓: {tiling_region_exteriors[0][1]}")
        if(len(tiles_exteriors_contour_list) != 1):
            print("樂高磚內部未相連！！！")
        else:
            time_start = time.time() #開始計時
            sim = similarity_compute(tiles_exteriors_contour_list[0][1], tiling_region_exteriors[0][1])
            #sim2 = similarity_compute2(tiles_exteriors_contour_list[0][1], tiling_region_exteriors[0][1])
            #sim3 = frechet_distance(tiles_exteriors_contour_list[0][1], tiling_region_exteriors[0][1])

            if(len(tiles_interiors_contour_list) != 0):
                for k in range(len(tiles_interiors_contour_list)):
                    sim  = sim  + similarity_compute(tiles_interiors_contour_list[k][1], tiling_region_interiors[k][1])
                    #sim2 = sim2 + similarity_compute2(tiles_interiors_contour_list[k][1], tiling_region_interiors[k][1])
                    #sim3 = sim3 + frechet_distance(tiles_interiors_contour_list[k][1], tiling_region_interiors[k][1])

                sim  = sim  / (len(tiles_interiors_contour_list) + 1)
                #sim2 = sim2 / (len(tiles_interiors_contour_list) + 1)
                #sim3 = sim3 / (len(tiles_interiors_contour_list) + 1)
            
            s = (0.3 + 0.25) * 2 #0.3為樂高磚高度,0.25為寬度
            fd  = (s - sim)  / s
            #fd2 = (s - sim2) / s
            #fd3 = (s - sim3) / s
            print("##################")
            print(f"similarity(論文): {fd}")   #輪廓擴展後分段計算
            #print(f"similarity2(未擴展): {fd2}")  #輪廓擴展後直接計算
            #print(f"similarity3(未分段): {fd3}")  #直接計算
            print("##################")
            time_end = time.time()    #結束計時
            time_c= time_end - time_start   #執行所花時間
            print('similarity-time cost', time_c, 's')
        #'''

        ##針對selected_tiles先處理斜磚合併並寫入，再寫入未合併的基本磚（因為斜磚合併會刪掉部份基本磚）
        ldr_x = 20
        ldr_y = 24
        slope_brick_end = False #判斷斜磚是否處理完畢
        restart = False #在第二圈處理基本磚時
        #lego_3044b_idx = [] #紀錄3040b的半斜磚索引值，待其餘斜磚處理完後，基本磚處理前進行處理
        slope_num = [] #紀錄斜磚的索引值
        slope_xy = []
        ff = open("unity_use/node_pos_test.txt",'w+')
        ff.write("0 (可讀入input名稱)" + "\n")
        i = 0
        # print("len(selected_tiles) = ", len(selected_tiles))
        # print("selected_tiles = ", selected_tiles)
        while i < len(selected_tiles):
            if(len(selected_tiles[i][1])-1 != 3 and slope_brick_end == False):
                if(i == len(selected_tiles)-1):
                    slope_brick_end = True
                    i = 0
                    continue
                i = i + 1
                continue
            elif(slope_brick_end == False):
                slope_num.append(i)
            else:
                for j in range(len(slope_num)):
                    if(i == slope_num[j]):
                        restart = True
                        break
            if(restart == True):
                restart = False
                i = i + 1
                continue
            average_x = 0.0
            average_y = 0.0
            max_x = 0.0
            min_x = 0.0
            max_y = 0.0
            min_y = 0.0
            corner_x = 0.0
            corner_y = 0.0
            temp = 0
            det_x = 0.0
            det_y = 0.0
            ## 位置（x,y,z）
            if(len(selected_tiles[i][1])-1 == 3):                                                  #對斜磚部份做個別的位移
                if(round(selected_tiles[i][1][0][0],2) != round(selected_tiles[i][1][1][0],2) and \
                   round(selected_tiles[i][1][0][1],2) != round(selected_tiles[i][1][1][1],2)):
                    corner_x = round(selected_tiles[i][1][2][0],2)
                    corner_y = -round(selected_tiles[i][1][2][1],2)
                    temp = 2
                elif(round(selected_tiles[i][1][1][0],2) != round(selected_tiles[i][1][2][0],2) and \
                     round(selected_tiles[i][1][1][1],2) != round(selected_tiles[i][1][2][1],2)):
                    corner_x = round(selected_tiles[i][1][0][0],2)
                    corner_y = -round(selected_tiles[i][1][0][1],2)
                    temp = 0
                else:
                    corner_x = round(selected_tiles[i][1][1][0],2)
                    corner_y = -round(selected_tiles[i][1][1][1],2)
                    temp = 1
            for j in range(len(selected_tiles[i][1])-1):
                if(j == 0):
                    max_x = round(selected_tiles[i][1][j][0],2)                       #x+向上
                    min_x = round(selected_tiles[i][1][j][0],2)
                    max_y = -round(selected_tiles[i][1][j][1],2)                      #對y座標加負號使y+向上
                    min_y = -round(selected_tiles[i][1][j][1],2)
                else:
                    if(max_x < round(selected_tiles[i][1][j][0],2)):  max_x = round(selected_tiles[i][1][j][0],2)
                    if(min_x > round(selected_tiles[i][1][j][0],2)):  min_x = round(selected_tiles[i][1][j][0],2)
                    if(max_y < -round(selected_tiles[i][1][j][1],2)): max_y = -round(selected_tiles[i][1][j][1],2)
                    if(min_y > -round(selected_tiles[i][1][j][1],2)): min_y = -round(selected_tiles[i][1][j][1],2)
                if(len(selected_tiles[i][1])-1 == 3):                                 #對斜磚部份做擺放的判斷
                    if(j == temp):
                        continue
                    elif(corner_x == round(selected_tiles[i][1][j][0],2)):
                        det_y = -round(selected_tiles[i][1][j][1],2) - corner_y
                    else:
                        det_x = round(selected_tiles[i][1][j][0],2) - corner_x
            ## 判斷三角形在旋轉和翻轉下的4種不同情形
            if(len(selected_tiles[i][1])-1 == 3):
                height = int(round((max_y - min_y)*80/ldr_y,1))   #判斷斜磚高度,以此排除跟height數量相同的基本磚
                # weight = int(round((max_x - min_x)*80/ldr_x,1))   #專門處理3044b雙斜磚
                ## 判斷是否為(1.樂高磚54200 / 2.樂高磚3044b雙斜磚 / 3.都不是)
                if(height == 0):
                    if(det_x > 0 and det_y > 0):                                                                                    #斜邊在右上,90度角在左下
                        average_x = round( ((max_x + min_x)/2) * 80, 1)
                        average_y = round( ((max_y + min_y)/2) * 80, 1) - round((max_y - min_y)/2 * 80,1) - ldr_y/2
                    elif(det_x < 0 and det_y > 0):                                                                                  #斜邊在左上,90度角在右下
                        average_x = round( ((max_x + min_x)/2) * 80, 1)
                        average_y = round( ((max_y + min_y)/2) * 80, 1) - round((max_y - min_y)/2 * 80,1) - ldr_y/2
                #elif(height == 1 and weight == 1):
                    
                else:
                    for k in range(height):
                        if(det_x > 0 and det_y > 0):                                                                                    #斜邊在右上,90度角在左下
                            average_x = round( ((max_x + min_x)/2) * 80, 1) - (20 + ((round((max_x - min_x) * 80,1) / ldr_x)-1)*ldr_x/2)
                            average_y = round( ((max_y + min_y)/2) * 80, 1) + (((round((max_y - min_y) * 80,1) / ldr_y)-1)*ldr_y/2) - k*ldr_y
                        elif(det_x < 0 and det_y > 0):                                                                                  #斜邊在左上,90度角在右下
                            average_x = round( ((max_x + min_x)/2) * 80, 1) + (20 + ((round((max_x - min_x) * 80,1) / ldr_x)-1)*ldr_x/2)
                            average_y = round( ((max_y + min_y)/2) * 80, 1) + (((round((max_y - min_y) * 80,1) / ldr_y)-1)*ldr_y/2) - k*ldr_y
                        elif(det_x > 0 and det_y < 0):                                                                                  #斜邊在右下,90度角在左上
                            average_x = round( ((max_x + min_x)/2) * 80, 1) - (20 + ((round((max_x - min_x) * 80,1) / ldr_x)-1)*ldr_x/2)
                            average_y = round( ((max_y + min_y)/2) * 80, 1) - (((round((max_y - min_y) * 80,1) / ldr_y)-1)*ldr_y/2) + k*ldr_y
                        else:                                                                                                           #斜邊在左下,90度角在右上
                            average_x = round( ((max_x + min_x)/2) * 80, 1) + (20 + ((round((max_x - min_x) * 80,1) / ldr_x)-1)*ldr_x/2)
                            average_y = round( ((max_y + min_y)/2) * 80, 1) - (((round((max_y - min_y) * 80,1) / ldr_y)-1)*ldr_y/2) + k*ldr_y
                        slope_xy.append([])
                        slope_xy[len(slope_xy)-1].append(average_x)
                        slope_xy[len(slope_xy)-1].append(average_y)
                        if(height != 1 and k == height-1):
                            if(det_y > 0):
                                average_y = average_y + k*ldr_y
                            else:
                                average_y = average_y - k*ldr_y

            else:
                average_x = round( ((max_x + min_x)/2) * 80, 1)
                average_y = round( ((max_y + min_y)/2) * 80, 1)
                ## 排除因斜磚合併而不該存在的基本磚3005
                for j in range(len(slope_xy)):
                    if([average_x,average_y] == slope_xy[j]):
                        restart = True
                        break
            if(restart == True):
                restart = False
                i = i + 1
                continue

            ff.write("1  " + str(color_num[i]) +" ")                                                     # str(color_num[i])為顏色編號
            ff.write(str(-average_x) + " " + str(-average_y) + " 0 ")                   #讀進unity須加入負號，使其從-z看向+z時是正確的
            
            # print(selected_feature)
            # print(len(selected_feature))

            #旋轉矩陣
            for k in range(len(selected_feature[i])-1):
                if(selected_feature[i][k] == 1):
                    node_kind = k
                    break
            if( node_kind == 0 or node_kind ==((len(selected_feature[i])-1)/2) ):
                ff.write("1 0 0 0 1 0 0 0 1 ")
            elif( 1 <= node_kind < ((len(selected_feature[i])-1)/2) ):
                if(det_y > 0):
                    ff.write("0 0 -1 0 1 0 1 0 0 ")
                else:
                    ff.write("0 0 1 0 1 0 -1 0 0 ")
            else:
                if(det_y > 0):
                    ff.write("0 0 1 0 1 0 -1 0 0 ")
                else:
                    ff.write("0 0 -1 0 1 0 1 0 0 ")
            
            ## 樂高磚種類
            # lego 2.2
            '''
            node_kind = node_kind % 2
            if(node_kind == 0):
                ff.write("3005.DAT")
            elif(node_kind == 1):
                if(det_y > 0):
                    ff.write("3040b.DAT")
                else:
                    ff.write("3665.DAT")
            '''
            # lego 4.1
            '''
            node_kind = node_kind % 3
            if(node_kind == 0):
                ff.write("3005.DAT")
            elif(node_kind == 1):
                if(det_y > 0):
                    ff.write("3040b.DAT")
                else:
                    ff.write("3665.DAT")
            elif(node_kind == 2):
                if(det_y > 0):
                    ff.write("4286.DAT")
                else:
                    ff.write("4287b.DAT")
            '''
            # lego 8.3
            #'''
            node_kind = node_kind % 4
            if(node_kind == 0):
                ff.write("3005.DAT")
            elif(node_kind == 1):
                if(det_y > 0):
                    ff.write("3040b.DAT")
                else:
                    ff.write("3665.DAT")
            elif(node_kind == 2):
                if(det_y > 0):
                    ff.write("4286.DAT")
                else:
                    ff.write("4287b.DAT")
            elif(node_kind == 3):
                ff.write("60481.DAT")     #不存在2x1x2的反斜磚 
            #'''
            # lego 10
            '''
            node_kind = node_kind % 5
            if(node_kind == 0):
                ff.write("3005.DAT")
            elif(node_kind == 1):
                if(det_y > 0):
                    ff.write("3040b.DAT")
                else:
                    ff.write("3665.DAT")
            elif(node_kind == 2):
                if(det_y > 0):
                    ff.write("4286.DAT")
                else:
                    ff.write("4287b.DAT")
            elif(node_kind == 3):
                ff.write("54200.DAT")
            elif(node_kind == 4):
                ff.write("60481.DAT")
            '''
            '''
            node_kind = node_kind % 5
            if(node_kind == 0):
                ff.write("3005.DAT")
            elif(node_kind == 1):
                if(det_y > 0):
                    ff.write("3040b.DAT")
                else:
                    ff.write("3665.DAT")
            elif(node_kind == 2):
                if(det_y > 0):
                    ff.write("4286.DAT")
                else:
                    ff.write("4287b.DAT")
            elif(node_kind == 3):
                if(det_y > 0):
                    ff.write("4460b.DAT")
                else:
                    ff.write("2449.DAT")
            elif(node_kind == 4):
                ff.write("60481.DAT")

            '''
            ff.write("\n")
            if(i == len(selected_tiles)-1 and slope_brick_end == False):
                slope_brick_end = True
                i = 0
                continue
            i = i + 1
        ff.close()

        plotter.draw_contours(debugger.file_path(file_name),
                              tiling_region_exteriors + tiling_region_interiors + super_contour_exteriors + super_contour_interiors + selected_tiles)
        if():
            return fd
        else:
            return 0
    def show_super_contour(self, plotter, debugger, file_name):
        super_contour_poly = self.get_super_contour_poly()
        exteriors_contour_list, interiors_list = BrickLayout.get_polygon_plot_attr(super_contour_poly, show_line = True)
        plotter.draw_contours(debugger.file_path(file_name), exteriors_contour_list + interiors_list)

    def show_adjacency_graph(self, save_path, edge_type="all", is_vis_prob=True, node_size=10,
                             edge_width=0.7, xlim=(-1, 1.6), ylim=(-1, 1.6)):
        # create Graph
        G_symmetric = nx.Graph()
        col_edges = [tuple(self.collide_edge_index[:, i]) for i in
                     range(self.collide_edge_index.shape[1])] if self.collide_edge_index.shape[
                                                                             0] > 0 else []
        adj_edges = [tuple(self.align_edge_index[:, i]) for i in
                     range(self.align_edge_index.shape[1])] if self.align_edge_index.shape[
                                                                           0] > 0 else []
        if edge_type == "all":
            edges = col_edges + adj_edges
        elif edge_type == "collision":
            edges = col_edges
        elif edge_type == "adjacent":
            edges = adj_edges
        else:
            print(f"error edge type!!! {edge_type}")

        edge_color = ["gray" for i in range(len(edges))]

        # draw networks
        G_symmetric.add_nodes_from(range(self.node_feature.shape[0]))
        node_color = [self.predict_probs[i] if is_vis_prob else "blue" for i in
                      range(self.node_feature.shape[0])]
        tile_indices = [self.inverse_index[i] for i in range(self.node_feature.shape[0])]
        node_pos_pts = [self.complete_graph.tiles[index].tile_poly.centroid for index in tile_indices]
        node_pos = list(map(lambda pt: [pt.x, - pt.y], node_pos_pts))
        #print(node_pos)

        vmin, vmax = 0.0, 1.0
        cmap = plt.cm.Reds
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        nx.draw_networkx(G_symmetric, pos=node_pos, node_size=node_size, node_color=node_color, cmap=cmap,
                         width=edge_width, edgelist=edges, edge_color=edge_color,
                         vmin=vmin, vmax=vmax, with_labels=False, style="dashed" if col_edges else "solid")

        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.savefig(save_path, dpi=400)
        print(f'saving file {save_path}...')
        plt.close()

    def show_predict_prob(self, plotter, debugger, file_name):
        # show prediction probs with color

        # predict_probs與node_color一樣
        predict_probs = self.predict_probs
        min_fill_color, max_fill_color = np.array([255,255,255,50]), np.array([255,0,0,50])
        min_pen_color, max_pen_color = np.array([255, 255, 255, 0]), np.array([127, 127, 127, 255])

        super_contour_poly = self.get_super_contour_poly()
        exteriors_contour_list, interiors_list = BrickLayout.get_polygon_plot_attr(super_contour_poly, show_line = True)

        #### sort by prob
        sorted_indices = np.argsort(self.predict_probs)

        plotter.draw_contours(debugger.file_path(file_name),
                              exteriors_contour_list + interiors_list + [self.complete_graph.tiles[self.inverse_index[i]].get_plot_attribute(
                                  (
                                      tuple(interp(predict_probs[i], vec1 = min_fill_color, vec2 = max_fill_color)),
                                      tuple(interp(predict_probs[i], vec1=min_pen_color, vec2=max_pen_color))
                                  )
                              ) for i in
                               sorted_indices])

    def get_super_contour_poly(self):
        ### return super contour poly if already calculated
        if self.super_contour_poly is None:
            tiles = self.complete_graph.tiles
            selected_indices = [k for k in self.re_index.keys()]
            selected_tiles = [tiles[s].tile_poly.buffer(1e-6) for s in selected_indices]
            total_polygon = unary_union(selected_tiles).simplify(1e-6)
            self.super_contour_poly = total_polygon
            return total_polygon
        else:
            return self.super_contour_poly

    @staticmethod
    def get_polygon_plot_attr(input_polygon, show_line = False, style = None):
        # return color plot attribute given a shapely polygon
        # return poly attribute with color
        
        exteriors_contour_list = []
        interiors_list = []

        ### set the color for exterior and interiors

        if style is None:
            color = 'light_gray_border' if show_line else 'light_gray'
        else:
            color = (style[0], (127, 127, 127, 0)) if show_line else style

        background_color = 'white_border' if show_line else 'white'

        if isinstance(input_polygon, shapely.geometry.polygon.Polygon):
            exteriors_contour_list = [(color, np.array(list(input_polygon.exterior.coords)))]
            interiors_list = [(background_color, np.array(list(interior_poly.coords))) for interior_poly in
                              input_polygon.interiors]

        elif isinstance(input_polygon, shapely.geometry.multipolygon.MultiPolygon):
            exteriors_contour_list = [(color, np.array(list(polygon.exterior.coords))) for polygon in input_polygon]
            for each_polygon in input_polygon:
                one_interiors_list = [(background_color, np.array(list(interior_poly.coords))) for interior_poly in
                                      each_polygon.interiors]
                interiors_list = interiors_list +  one_interiors_list

        return exteriors_contour_list, interiors_list

    def get_selected_tiles(self):
        return [self.complete_graph.tiles[self.inverse_index[i]].tile_poly for i in range(len(self.predict)) if self.predict[i] == 1]

    def get_selected_tiles_union_polygon(self):
        return unary_union(self.get_selected_tiles())

    def detect_holes(self):
        ### DETECT HOLE
        selected_tiles = [self.complete_graph.tiles[self.inverse_index[i]].tile_poly.buffer(1e-7) for i in range(len(self.predict)) if self.predict[i] == 1]
        unioned_shape = unary_union(selected_tiles)
        if isinstance(unioned_shape, shapely.geometry.polygon.Polygon):
            if len(list(unioned_shape.interiors)) > 0:
                return True
        elif isinstance(unioned_shape, shapely.geometry.multipolygon.MultiPolygon):
            if any([len(list(unioned_shape[i].interiors)) > 0 for i in range(len(unioned_shape))]):
                return True

        return False

    def get_data_as_torch_tensor(self, device):
        x, adj_edge_index, adj_edge_features, collide_edge_index, collide_edge_features = \
            util.data_util.to_torch_tensor(device, self.node_feature, self.align_edge_index, self.align_edge_features, self.collide_edge_index, self.collide_edge_features)

        return x, adj_edge_index, adj_edge_features, collide_edge_index, collide_edge_features
    
    def compute_sub_layout(self, predict):
        assert len(self.node_feature) == len(predict.labelled_nodes) + len(predict.unlabelled_nodes)
        sorted_dict = sorted(predict.unlabelled_nodes.items(), key=lambda x: x[0])
        predict.unlabelled_nodes.clear()
        predict.unlabelled_nodes.update(sorted_dict)
        # compute index mapping from original index to current index
        node_re_index = {}
        for idx, key in enumerate(predict.unlabelled_nodes):
            node_re_index[key] = idx

        complete_graph = self.complete_graph
        node_feature = self.node_feature[list(predict.unlabelled_nodes.keys())]

        # index
        collide_edge_index = [ [node_re_index[self.collide_edge_index[0, i]], node_re_index[self.collide_edge_index[1, i]]] for i in range(self.collide_edge_index.shape[1]) if self.collide_edge_index[0, i] in predict.unlabelled_nodes and self.collide_edge_index[1, i] in predict.unlabelled_nodes ] \
            if self.collide_edge_index.shape[0] > 0 else np.array([])
        collide_edge_index = np.array(collide_edge_index).T

        align_edge_index = [ [node_re_index[self.align_edge_index[0, i]], node_re_index[self.align_edge_index[1, i]]] for i in range(self.align_edge_index.shape[1]) if self.align_edge_index[0, i] in predict.unlabelled_nodes and self.align_edge_index[1, i] in predict.unlabelled_nodes ] \
            if self.align_edge_index.shape[0] > 0 else np.array([])
        align_edge_index = np.array(align_edge_index).T

        # feature
        collide_edge_features = np.array([ self.collide_edge_features[i, :] for i in range(self.collide_edge_index.shape[1]) if self.collide_edge_index[0, i] in predict.unlabelled_nodes and self.collide_edge_index[1, i] in predict.unlabelled_nodes ]) \
            if self.collide_edge_features.shape[0] > 0 else np.array([])
        align_edge_features = np.array([ self.align_edge_features[i, :] for i in range(self.align_edge_index.shape[1]) if self.align_edge_index[0, i] in predict.unlabelled_nodes and self.align_edge_index[1, i] in predict.unlabelled_nodes ]) \
            if self.align_edge_features.shape[0] > 0 else np.array([])


        # compute index mapping from current index to original index
        node_inverse_index = {}
        for idx, key in enumerate(predict.unlabelled_nodes):
            node_inverse_index[idx] = key

        fixed_re_index = {}
        for i in range(node_feature.shape[0]):
            fixed_re_index[self.inverse_index[node_inverse_index[i]]] = i

        return BrickLayout(complete_graph, node_feature, collide_edge_index, collide_edge_features, align_edge_index, align_edge_features, fixed_re_index, target_polygon=self.target_polygon), node_inverse_index

    @staticmethod
    def assert_equal_layout(brick_layout_1, brick_layout_2):
        assert np.array_equal(brick_layout_1.node_feature, brick_layout_2.node_feature)
        assert np.array_equal(brick_layout_1.collide_edge_index, brick_layout_2.collide_edge_index)
        assert np.array_equal(brick_layout_1.collide_edge_features, brick_layout_2.collide_edge_features)
        assert np.array_equal(brick_layout_1.align_edge_index, brick_layout_2.align_edge_index)
        assert np.array_equal(brick_layout_1.align_edge_features, brick_layout_2.align_edge_features)

        # mapping from index of complete graph to index of super graph
        for key in brick_layout_1.re_index.keys():
            assert brick_layout_1.re_index[key] == brick_layout_2.re_index[key]
        for key in brick_layout_2.re_index.keys():
            assert brick_layout_2.re_index[key] == brick_layout_1.re_index[key]

        ### assert others
        assert np.array_equal(brick_layout_1.predict, brick_layout_2.predict)
        assert np.array_equal(brick_layout_1.predict_probs, brick_layout_2.predict_probs)
        assert brick_layout_1.predict_order == brick_layout_2.predict_order
        assert brick_layout_1.target_polygon == brick_layout_2.target_polygon

if __name__ == "__main__":
    pass
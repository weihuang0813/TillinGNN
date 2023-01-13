import torch
import numpy as np
import time
import inputs.config as config
import math
import itertools
import traceback
from graph_networks.network_utils import get_network_prediction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = 1e-7

class Losses:
    # evaluate loss for a given data set
    @staticmethod
    def cal_avg_loss(network, data_set_loader):
        losses = []
        avg_collision_probs_list = []
        filled_area_list = []
        avg_align_length_list = []
        for batch in data_set_loader:

            try:
                data = batch.to(device)
                probs = get_network_prediction(
                    network=network,
                    x=data.x,
                    adj_e_index=data.edge_index,
                    adj_e_features=data.edge_features,
                    col_e_idx=data.collide_edge_index,
                    col_e_features=None)

                if probs is None:
                    continue

                loss, min_index, _ = Losses.calculate_unsupervised_loss(probs, data.x, data.collide_edge_index,
                                                                        adj_edges_index=data.edge_index,
                                                                        adj_edge_features=data.edge_features)

                losses.append(loss.detach().cpu().numpy())
            except:
                print(traceback.format_exc())

        return np.mean((losses)), np.mean((avg_collision_probs_list)), np.mean((filled_area_list)), np.mean(
            (avg_align_length_list))


    @staticmethod
    def calculate_unsupervised_loss(probs, node_feature, collide_edge_index, adj_edges_index, adj_edge_features):
        # probs 為N*1的二維陣列(值是介於0到1)
        # print(probs)
        # print(len(probs))
        # node_feature 為N*(2*input_num + 1)的二維陣列(最後一個是標準化後的面積比,前面其他則是以one-hot紀錄各圖塊,包含旋轉後為新圖塊)
        #print(node_feature)

        # start time
        start_time = time.time()
        N = probs.shape[0]  # number of nodes
        M = probs.shape[1]  # number of output features
        E_col = collide_edge_index.shape[1] if len(collide_edge_index) > 0 else 0  # to handle corner cases when no collision edge exist
        E_adj = adj_edges_index.shape[1] if len(adj_edges_index) > 0 else 0
        print(E_adj)
        losses = []
        COLLISION_WEIGHT    = config.COLLISION_WEIGHT
        ALIGN_LENGTH_WEIGHT = config.ALIGN_LENGTH_WEIGHT
        AVG_AREA_WEIGHT     = config.AVG_AREA_WEIGHT
        
        NODE_WEIGHT         = config.NODE_WEIGHT
        #CONNECT_WEIGHT      = config.CONNECT_WEIGHT
        
        #print( COLLISION_WEIGHT, ',', ALIGN_LENGTH_WEIGHT, ',', AVG_AREA_WEIGHT, ',', NODE_WEIGHT)

        for sol in range(M):
            solution_prob = probs[:, sol] # 一維陣列N個元素
            print(len(solution_prob))
            # print(solution_prob)

            ########### average node area loss
            avg_tile_area = torch.clamp(torch.mean(node_feature[:, -1] * solution_prob), min=eps) # node_feature[:, -1] 代表每列的倒數第一個元素...共N個
            loss_ave_area = torch.log(avg_tile_area)

            ########### collision feasibility loss
            if E_col > 0:
                first_index = collide_edge_index[0, :]
                first_prob = torch.gather(solution_prob, dim=0, index=first_index)
                second_index = collide_edge_index[1, :]
                second_prob = torch.gather(solution_prob, dim=0, index=second_index)

                prob_product = torch.clamp(first_prob * second_prob, min=eps, max=1 - eps)
                loss_per_edge = torch.log(1 - prob_product)
                loss_per_edge = loss_per_edge.view(-1)
                loss_feasibility = torch.sum(loss_per_edge) / E_col
            else:
                loss_feasibility = 0.0

            ########### edge length loss
            if E_adj > 0:
                adj_edge_lengths = adj_edge_features[:, 1]

                first_index = adj_edges_index[0, :]
                first_prob = torch.gather(solution_prob, dim=0, index=first_index)
                second_index = adj_edges_index[1, :]
                second_prob = torch.gather(solution_prob, dim=0, index=second_index)
                assert (first_prob * second_prob * adj_edge_lengths >= 0).all() or not (first_prob * second_prob * adj_edge_lengths <= 1).all()

                prob_product = torch.clamp(first_prob * second_prob * adj_edge_lengths, min=eps)
                loss_per_adjedge = torch.log(prob_product) / math.log(10)
                loss_per_adjedge = loss_per_adjedge.view(-1)
                loss_align_length = torch.sum(loss_per_adjedge) / E_adj
            else:
                loss_align_length = 0.0
            
            ########### lego loss

            kk = 0
            for i in range(len(solution_prob)):
                if(node_feature[i][0] == 1):
                    kk = kk + 1
            if(kk == 0):
                kk = 0.1
            loss_ave_node = math.log(kk/len(solution_prob))
            '''
            prob_sum = torch.sum(solution_prob)
            prob_lego = torch.sum(solution_prob)
            for i in range(len(solution_prob)):
                if(node_feature[i][0] != 1):
                    prob_lego = torch.sub(prob_lego, solution_prob[i])
            if(prob_lego == prob_sum):
                prob_lego = 0.1
            loss_ave_node = torch.log(prob_lego/prob_sum)
            '''
            '''
            ########### connection loss
            if E_adj > 0:
            
                #first_index = adj_edges_index[0, :]
                #first_prob = torch.gather(solution_prob, dim=0, index=first_index)
                #second_index = adj_edges_index[1, :]
                #second_prob = torch.gather(solution_prob, dim=0, index=second_index)
                #prob_product = torch.clamp(first_prob * second_prob, min=eps)
                #print(prob_product)
                #sum_product = torch.sum(prob_product) / len(solution_prob)
                #print(sum_product)  
            
                sum_product = E_adj / (len(solution_prob)*4)
                print(sum_product)

                loss_connection = math.log(sum_product)
                print(loss_connection)
                print("//////////////////////")
            else:
                loss_connection = 0.0
            '''

            assert loss_feasibility <= 0, 'false1'
            assert loss_ave_area <= 0, 'false2'
            assert loss_align_length <= 0, 'false3'
            
            assert loss_ave_node <= 0, 'false4'
            #assert loss_connection <= 0, 'false5'

            #AVG_AREA_WEIGHT = 0
            #COLLISION_WEIGHT = 0
            #ALIGN_LENGTH_WEIGHT = 0
            #NODE_WEIGHT = 0
            #CONNECT_WEIGHT=0
            
            test1 = 1 - AVG_AREA_WEIGHT     * loss_ave_area
            test2 = 1 - COLLISION_WEIGHT    * loss_feasibility
            test3 = 1 - ALIGN_LENGTH_WEIGHT * loss_align_length
            
            test4 = 1 - NODE_WEIGHT         * loss_ave_node
            #test5 = 1 - CONNECT_WEIGHT      * loss_connection
            
            loss = (1 - AVG_AREA_WEIGHT     * loss_ave_area   ) * \
                   (1 - COLLISION_WEIGHT    * loss_feasibility) * \
                   (1 - ALIGN_LENGTH_WEIGHT * loss_align_length)* \
                   (1 - NODE_WEIGHT         * loss_ave_node)    #* \
                   #(1 - CONNECT_WEIGHT      * loss_connection)
            '''

            loss = (1 - AVG_AREA_WEIGHT     * loss_ave_area   ) * \
                   (1 - COLLISION_WEIGHT    * loss_feasibility) * \
                   (1 - ALIGN_LENGTH_WEIGHT * loss_align_length)
            '''
            
            #print( test1, ',', test2, ',', test3, ',', test4, ',', test5, ',node_number:', N)
            print('loss:', loss)
            assert loss >= 1.0, 'loss 為負的'

            losses.append(loss)

        losses = torch.stack(losses)
        loss = torch.min(losses)

        # print(f"unsupverised loss : {loss}, time_used = {time.time() - start_time}")
        min_index = torch.argmin(losses).detach().cpu().numpy()
        losses = losses.detach().cpu().numpy()
        return loss, min_index, losses

    # to evaluate the quality of a collision-free solution
    @staticmethod
    def solution_score(predict, brick_layout):
        predict = torch.from_numpy(np.array(predict)).float().to(device)
        x, adj_edge_index, adj_edge_features, collide_edge_index, collide_edge_features = \
            brick_layout.get_data_as_torch_tensor(device)

        ######## calculate total area
        filled_area = predict.dot(x[:,-1]*brick_layout.complete_graph.max_area) / brick_layout.get_super_contour_poly().area
        assert filled_area >= -1e-7 and filled_area <= 1+1e-7

        ######## calculate alignment length
        if adj_edge_features.size(0) > 0:
            adj_edge_lengths = adj_edge_features[:, 1] * brick_layout.complete_graph.max_align_length
            first_index = adj_edge_index[0, :]
            first_prob = torch.gather(predict, dim=0, index=first_index)
            second_index = adj_edge_index[1, :]
            second_prob = torch.gather(predict, dim=0, index=second_index)
            if not (first_prob * second_prob * adj_edge_lengths >= 0).all() or not (
                    first_prob * second_prob * adj_edge_lengths <= brick_layout.complete_graph.max_align_length).all():
                input()
            loss_align_length = (first_prob * second_prob).dot(adj_edge_lengths) # total edge length
        else:
            loss_align_length = 0.0

        all_edge_length = sum([brick_layout.complete_graph.tiles[brick_layout.inverse_index[i]].get_perimeter() for i in range(len(predict))
         if predict[i] == 1])

        assert (loss_align_length/all_edge_length) > -1e-7 and (loss_align_length/all_edge_length) < 1+1e-7

        return (config.AVG_AREA_WEIGHT * filled_area + config.ALIGN_LENGTH_WEIGHT * (loss_align_length/all_edge_length)).detach().cpu().item()





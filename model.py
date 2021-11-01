import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F
import pandas as pd
#from optimization import Adam

import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax













###########################################################################





class MAPredNet(nn.Module):
    def __init__(self):
        super(MAPredNet, self).__init__()
        # hyperparams
        self.s_year = 1997
        self.e_year = 2020
        self.a_freq_fv_dim = 14
        self.target_fv_dim = 17
        self.embedding_b = 32
        self.embedding_c = 16
        self.embedding_z = 32
        self.dropout_ratio = 0.25
        self.device =   torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        # define models
        self.timing_net = Timing_Net(self.embedding_b, self.embedding_c)
        # # input: (L1, a_freq_fv_dim); output: (L1, embedding_b)
        self.b_net = nn.Sequential(
                        nn.Linear(in_features=self.a_freq_fv_dim, out_features=64), 
                        #nn.Dropout(self.dropout_ratio), 
                        nn.LeakyReLU(),
                        nn.Linear(in_features=64, out_features=self.embedding_b))
        self.c_net =  nn.GRU(input_size = 3, hidden_size = self.embedding_c, batch_first=True)
        self.choice_net = ChoiceNet(self.embedding_b, self.embedding_c, self.embedding_z)




    def MCestimator(self, arr, estimate_length):
        '''
        Monte Carlo Estimator 
        input: 
            arr: 1d arr
            estimate_length: scalar
        '''
        estimation =  estimate_length*(1/(arr.size(0)))*torch.sum(arr)
        
        return estimation

    def likelihood(self, event_time_ll, non_event_time_ll):
        '''
        compute likelihood loss
        input are all scalars
        loss: small = good
        '''
        loss = - event_time_ll + non_event_time_ll
        return loss

    def forward(self, arr_b, arr_c, arr_delta_time, event_data, non_event_data, estimate_length, choice_data_dict):
        '''
        WARNING: 
            1. This Version only works for batch_size = 1 (for a single firm)..... (if batch size > 1, have to padding at first to make arr_b, arr_c has same size
                    if padding, the number of event of every year for all firms should be the same (in dict_idx[year]))
            2. all idx must be put into list in time order
            3. year is always an integer

        Assume:
        L1 = Length_of_year_cross
        L2 = length_of_self+peer_events
        L3 = length_of_self event
        L_Neg = Length of negative samples

        N_i_2 = N_self_event_ith_year


        Inputs:
            arr_b: 2d tensor; raw financial variables; (L1, 14) 

            arr_c: 2d tensor; raw peer/self effect variables; (L2, 3)

            arr_delta_time: 1d tensor; (L3, 1); 
                arr_delta_time is corresponding to arr_c: time delta in i row = (t_event_i+1 - t_event_i) 


            event_data: 
                arr_b_idx: 1d tensor [3, 3, 4, 4, 4, 5, 9 ...]
                    length = L3
                    element: integer as row number in arr_b; for true event
                arr_c_idx: 1d tensor, [3, 3, 4, 4, 4, 5, 9 ...]
                    length = L3
                    element: integer as row number in arr_c; for true event
                arr_delta_time:  (same thing)

                    

                    

                    

            
            non_event_data: for negative sampling in timing model only; idea: draw time point from Unif(0, MAX_T). 
                                pick up the corresponding b, c, delta_t.
                arr_b_idx:  1d tensor, 
                    length: L_Neg
                arr_c_idx: 1d tensor
                arr_t_non: 1d tensor
                

            estimate_length: scalar, for negative sampling MC estimator, 
                max(time)  - min(time)


            choice_data_dict: a dict, every year, output those 5 things
                   
                - dict_idx : each year has a list of 2 element
                    - arr_b_idx_i: lst, N_i_2 
                    - arr_c_idx_i: lst, N_i_2 
                
                - true_tar_idxs_i: each year has a torch tensor, one-hot, size = (N_i_2, N_i_1) # for each row, only 1 (true acquirer) is 1, others are 0.
                        N_i_1 = N_i (every firm could be in target candidate)
                - node features each year has a  2d tensor: [N_i, in_channels_i]  # inchannels = 
                - network structure : each year has a 2d tensor: [2, N_edges_i] # the idx here is corresponding to node feature array

                
        output:
        
        '''
        # check input data 
        # arr_b_idx, arr_c_idx, arr_delta_time = event_data # expand
        # arr_b_idx_non, arr_c_idx_non, arr_delta_time_non = non_event_data
        # assert (len(arr_b_idx) == len(arr_c_idx)), "the size of input indeces dismatch for event data"
        # assert (len(arr_b_idx_non) == len(arr_c_idx_non)), "the size of input indeces dismatch for non-event data"
        # assert (arr_c.size[0] == arr_delta_time.size[0]), "the size of input array dismatch"


        # transform to embedding
        #arr_b = arr_b.float()
        #print("#### arr_b", arr_b, arr_b.size())
        mat_b = self.b_net(arr_b) # (L1, embedding_b)
        #print("#### mat_b",mat_b, mat_b.size())
    
        
        mat_c = self.c_net(arr_c)[0] # (B, L2, embedding_c); 
        #print("#### arr_b:", arr_b)

        # timing model

        event_lambdas = self.timing_net(mat_b, mat_c, event_data) # (L3, )
        #print(event_lambda)
        non_event_lambdas = self.timing_net(mat_b, mat_c, non_event_data) # (L_Neg, )
        #non_event_tuple = (non_event_lambdas, estimate_length)
        print("### event lambdas: ", event_lambdas) # any dimension can never have zero! other wise log(0) = -inf
        print("### non event lambdas: ",non_event_lambdas)

        event_time_ll = torch.sum(torch.log(event_lambdas))  # out = scalar
        
        non_event_time_ll = self.MCestimator(non_event_lambdas, estimate_length)  # out = scalar

        # choice model
        choice_l = self.choice_net(mat_b, mat_c, choice_data_dict, self.s_year, self.e_year)
        
        event_time_ll = event_time_ll 
        non_event_time_ll = non_event_time_ll
        choice_l = choice_l 
        
        
        print("##### event loss:", - event_time_ll.item(), "non event loss: ", non_event_time_ll.item(), "chocie_l:", choice_l.item() )

        total_l = - event_time_ll + choice_l + non_event_time_ll 
        return total_l, - event_time_ll , non_event_time_ll, choice_l




class Time_Transfer(nn.Module):
    def __init__(self):
        super(Time_Transfer, self).__init__()
        self.lin1a = nn.Linear(in_features=1, out_features=20)
        self.lin1b = nn.Linear(in_features=16, out_features=20)
        self.lin2 = nn.Linear(in_features=20, out_features=1)
       
    def forward(self, x, b):
        x1a = self.lin1a(x)
        x1b = self.lin1b(b)
        xa = torch.sigmoid(x1a+x1b)
        #x = torch.cat((xa, xb), dim=-1)
        x = self.lin2(xa)
        
        return x #.squeeze(dim=-1)


class Timing_Net(nn.Module):
    def __init__(self, embedding_b, embedding_c):
        super(Timing_Net, self).__init__()
        #self.phi =  torch.abs(torch.randn(1, requires_grad=True)) + 0.1
        self.base_rate = torch.randn(1, requires_grad=True)
        self.w_b =  torch.randn(embedding_b, requires_grad=True) # (embedding_b, )
        #self.w_b = w_b.unsqueeze(dim=1)
        self.w_c =  torch.randn(embedding_c, requires_grad=True)# (embedding_c, )
        #self.w_c = w_c.unsqueeze(dim=1)
        #self.omega = torch.randn(1, requires_grad=True)# scalar
        #self.event_data = event_data
        self.exponential_decay = Time_Transfer()
        
        # nn.Sequential(
        #                 nn.Linear(in_features=1, out_features=10),
        #                 nn.Linear(in_features=10, out_features=1))
        
        self.f_lambda = nn.Softplus(beta=1)
        self.device =   torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #self.season = torch.randn(embedding_b, requires_grad=True)
    def forward(self, mat_b, mat_c,  event_data):
        '''
        Assume:
            L1 = Length_of_year_cross
            L2: length_of_self+peer_events
            L3: length_of_self event
            
        Input: 
            mat_b: (L1, embedding_b)
            mat_c: (L2, embedding_c)
            arr_delta_time: (L2, embedding_c)

        
        '''
        arr_b_idx, arr_c_idx, arr_delta_t = event_data # expand


        b = torch.index_select(mat_b, 1, arr_b_idx[0]) # (L3, embedding_b) 
        print(b.get_device())

        c = torch.index_select(mat_c, 1, arr_c_idx[0]) # (L3, embedding_c)
        delta_t = arr_delta_t # (L3, 1)
        print("##### delta t:", delta_t)
        delta_t = delta_t.float()

        #decay_input = torch.cat((torch.transpose(delta_t, dim0=0, dim1=1), ))
        #print(delta_t.size(), b.size())
        t_emb = self.exponential_decay(delta_t.unsqueeze(-1), c) # B, L3
       

        #print("##### : ", t_emb.size(), b.size())
        # b: (B, L_3, embed), w_b : (embed)
        # when init, b: ~ -10^5; c: ~ - 1 omega: ~ 10^5--10^200
        
        rate =   torch.einsum('ble, e->bl', b, self.w_b)  + torch.einsum('ble, e->bl', c, self.w_c) +  t_emb.squeeze(-1) # torch.transpose(t_emb, dim0=0, dim1=1)#self.omega * torch.exp(-self.omega * delta_t) 
        #print(torch.einsum('ble, e->bl', b, self.w_b).size(), rate.size())
        # rate: (B, L_3)
        #print("### b:", b)
        # print(torch.einsum('ble, e->bl', b, self.w_b))
        # print(torch.einsum('ble, e->bl', c, self.w_c))
        # print(rate)
        #print("### rate:", rate)
        #lambda_dt = torch.log(rate+1)  # (L3, )
        #print(rate.size())
        #lambda_dt = self.f_lambda2(torch.transpose(rate, dim0=0, dim1=1))
        lambda_dt = self.f_lambda(rate)
        #print(lambda_dt)
        return   lambda_dt #+ 1e-10









class ChoiceNet(nn.Module):
    
    def __init__(self, embedding_b, embedding_c, embedding_z):
        super(ChoiceNet, self).__init__()
        # hyperparams
        self.embedding_z = embedding_z
        self.embedding_b, self.embedding_c = embedding_b, embedding_c
        self.gnn_hidden_dim = 64
        self.gnn_model_type = "GraphSage" # GraphSage or GAT
        self.target_fv_dim = 17
        self.dropout_ratio = 0.25

        # build modules
        self.gnn_choice = GNN_Stack(self.target_fv_dim, self.gnn_hidden_dim, self.embedding_z, self.gnn_model_type)
        self.transform = nn.Sequential(  # input dim = (N_of_self_event, embedding_b + embedding_c), output dim = (N_of_self_event, embedding_z)
                        nn.Linear(in_features=self.embedding_b + self.embedding_c, out_features=64), nn.Dropout(self.dropout_ratio), nn.ReLU(),
                        nn.Linear(in_features=64, out_features=self.embedding_z))
        self.loss = nn.BCELoss()
    
    #def list_of_tensor_to_tensor(lst_of_tensor):
        


    def forward(self, mat_b, mat_c, choice_data_dict, s_year, e_year):
        '''

        '''
        pred_logits_lst = []
        true_lst = []
        choice_loss_y_lst = []
        dict_idx, true_tar_idxs, features, edges = choice_data_dict
        for year in range(s_year, e_year+1):

            '''
            At the ith iteration of the loop,
                compute binary cross entropy loss for choice problem of i-th year
                based on all of the MA event occurred in that year
            
            N_i_1 = number of candidate target in i-th year
            N_i_2 = number of self events in i-th year

            '''
            dict_idx_i, true_tar_idxs_i,features_i, edges_i  = dict_idx[year], true_tar_idxs[year],features[year],edges[year] # list=:[N_i_1], arrays: (N_i_1, 22) , (2, |E|)
            arr_b_idx_i, arr_c_idx_i = dict_idx_i 

            # true_tar_idxs_i: tensor, one-hot, size = (N_i_2, N_i_1)
            # arr_b_idx_i, arr_c_idx_i: list: length = N_i_2 and N_i_2 
            # if there's no self event in ith year, continue
            #if arr_b_idx_i.size()[1] == 0: 
            #print("### arr_b_idx_i", arr_b_idx_i.shape)
            if arr_b_idx_i.size()[1] == 0:
            #if len(arr_b_idx_i) == 0:
                continue
            else:
                arr_b_idx_i = arr_b_idx_i.squeeze()
                arr_c_idx_i = arr_c_idx_i.squeeze()
                #print("### arr_b_idx_i", arr_b_idx_i)
                # arr_b_idx_i = torch.stack(arr_b_idx_i).squeeze()
                # arr_c_idx_i = torch.stack(arr_c_idx_i).squeeze()

                # GNN part
                '''
                always pass the entire graph for i-th year into GNN
                '''
                
                #assert len(true_tar_idxs_i) == features_i.size(0), "number of self events mismatch in choice data"
                #print(edges_i.size())
                z_vt_i = self.gnn_choice(features_i.squeeze(), edges_i.squeeze()) # (N_i_1, embedding_z)

                # z_dt : (N_i_2, embedding_z)
                #print(mat_b)
                z_dt_i = self.transform(torch.cat((torch.index_select(mat_b, 1, arr_b_idx_i), torch.index_select(mat_c, 1, arr_c_idx_i)), dim=2))  # z_dt : (N_i_2, embedding_b + embedding_c)
                #print(z_dt_i)
                # broadcasting
                z_vt_i = z_vt_i.unsqueeze(0).unsqueeze(1) # (1, 1,  N_i_1, embedding_z) # add batch size back
                z_dt_i = z_dt_i.unsqueeze(2) # (1, N_i_2, 1,  embedding_z)

                logits_i = (z_dt_i * z_vt_i).sum(axis=-1) # (1, N_i_2, N_i_1)
                
                true_tar_idxs_i = true_tar_idxs_i.float()
                #print(torch.max(torch.sigmoid(logits_i)))
                choice_l = F.binary_cross_entropy(torch.sigmoid(logits_i), true_tar_idxs_i)  # inputs are both (N_i_2, N_i_1)
                #pred_logits_lst.append(logits_i) # (N_i_2, N_i_1)
                #true_lst.append(true_tar_idxs_i) # (N_i_2, N_i_1)
                choice_loss_y_lst.append(choice_l)
        #pred_logits = torch.stack(pred_logits_lst)
        #true_target = torch.stack(true_lst)
        choice_l = torch.stack(choice_loss_y_lst).sum()

        return choice_l#(pred_logits, true_target)


            
            





















                
##################### GNN ###########################################
            


class GNN_Stack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, model_type, emb=True):
        super(GNN_Stack, self).__init__()

        # arguments
        self.model_type = model_type
        self.num_layers = 2
        self.heads = 1
        self.dropout = 0.25
        





        conv_model = self.build_conv_model(self.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (self.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(self.num_layers-1):
            self.convs.append(conv_model(self.heads * hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(self.heads * hidden_dim, hidden_dim), nn.Dropout(self.dropout), 
            nn.Linear(hidden_dim, output_dim))

        self.dropout = self.dropout
        self.num_layers = self.num_layers

        self.emb = emb

    def build_conv_model(self, model_type):
        if model_type == 'GraphSage':
            return GraphSage 
        elif model_type == 'GAT':
            return GAT

    def forward(self, X, E):
        '''
        X: the node features: [N, input_dim]
        E: the network structure: [2, E]  # note that the idx is corresponding to X


        tar_net_fv_i, tar_net_E_i
        
        '''
        x, edge_index = X, E 
          
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)

        x = self.post_mp(x)

        if self.emb == True:
            return x

        return F.log_softmax(x, dim=1)


class GraphSage(MessagePassing):
    
    def __init__(self, in_channels, out_channels, normalize = True,
                 bias = False, **kwargs):  
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_l = nn.Linear(self.in_channels, self.out_channels)
        self.lin_r = nn.Linear(self.in_channels, self.out_channels)

        

        self.reset_parameters()


    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size = None):
        prop = self.propagate(edge_index, x=(x, x), size=size)
        x = x.float()
        prop  = prop.float()
        out = self.lin_l(x) + self.lin_r(prop)
        if self.normalize:
            out = F.normalize(out, p=2)

        return out

    def message(self, x_j):
        out = x_j
        return out

    def aggregate(self, inputs, index, dim_size = None):
        # The axis along which to index number of nodes.
        node_dim = self.node_dim
        out = torch_scatter.scatter(inputs, index, node_dim, dim_size=dim_size, reduce='mean')

        return out







class GAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads = 2,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = None
        self.lin_r = None
        self.att_l = None
        self.att_r = None


        # self.lin_l is the linear transformation that you apply to embeddings 
        self.lin_l = nn.Linear(self.in_channels, self.out_channels * self.heads)
        self.lin_r = self.lin_l

        self.att_l = nn.Parameter(torch.zeros(self.heads, self.out_channels))
        self.att_r = nn.Parameter(torch.zeros(self.heads, self.out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size = None):
        
        H, C = self.heads, self.out_channels


        x_l = self.lin_l(x).reshape(-1, H, C)
        x_r = self.lin_r(x).reshape(-1, H, C)
        alpha_l = self.att_l * x_l
        alpha_r = self.att_r * x_r
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size)
        out = out.reshape(-1, H*C)

        return out




    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):

        alpha = F.leaky_relu(alpha_i + alpha_j, negative_slope=self.negative_slope)
        if ptr:
            att_weight = F.softmax(alpha_i + alpha_j, ptr)
        else:
            att_weight = pyg.utils.softmax(alpha_i + alpha_j, index)
        att_weight = F.dropout(att_weight, p=self.dropout)
        out = att_weight * x_j


        return out


    def aggregate(self, inputs, index, dim_size = None):
        out = torch_scatter.scatter(inputs, index, self.node_dim, dim_size=dim_size, reduce='sum')
    
        return out



    


        
        
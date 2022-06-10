# Our implementation of Voting module
# CS492 ML for 3D data
# by Team 10 - Ali Ahmed and Rakhman Ulzhalgas


import torch
import torch.nn as nn
import torch.nn.functional as F
import keras
import keras.backend as K
import keras.layers as layers

class VotingModule(nn.Module):
    def __init__(self, factor_v, dim_feature):

        super().__init__()
        self.dim_in = dim_feature
        self.factor_v = factor_v
        self.dim_out = self.dim_in
        self.dim_o  = 3+self.dim_out
        dim_of = self.dim_o * self.factor_v
        self.c1 = torch.nn.Conv1d(self.dim_in, self.dim_in, 1)
        self.c2 = torch.nn.Conv1d(self.dim_in, self.dim_in, 1)
        self.c3 = torch.nn.Conv1d(self.dim_in, dim_of, 1)
        self.batch1 = torch.nn.BatchNorm1d(self.dim_in)
        self.batch2 = torch.nn.BatchNorm1d(self.dim_in)
        self.batch3 = torch.nn.BatchNorm1d(self.dim_in)

    def forward(self, s_tens, feat_s):
        num_seed = s_tens.shape[1]
        num_vote = num_seed*self.factor_v
        batch_size = s_tens.shape[0]
        net = F.relu(self.batch1(self.c1(feat_s))) 
        net = F.relu(self.batch2(self.c2(net))) 
        net = self.c3(net) 
        n_transpose = net.transpose(2,1)        
        net = n_transpose.view(batch_size, num_seed, self.factor_v, self.dim_o)
        vote_xyz = s_tens.unsqueeze(2) + net[:,:,:,0:3]
        vote_xyz = vote_xyz.contiguous().view(batch_size, num_vote, 3)
        f_transpose = feat_s.transpose(2,1)
        vote_features = f_transpose.unsqueeze(2) + net[:,:,:,3:]
        vote_features = vote_features.contiguous().view(batch_size, num_vote, self.dim_out)
        vote_features = vote_features.transpose(2,1).contiguous()
        
        return vote_xyz, vote_features
 
if __name__=='__main__':
    v_net = VotingModule(2, 256).cuda()
    seed, features = v_net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda())
    print('This is xyz:', seed.shape)
    print('These are features:', features.shape)

# Our implementation of Votenet system to generate proposals  
# CS492 ML for 3D data
# by Team 10 - Ali Ahmed and Rakhman Ulzhalgas


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import keras
import keras.backend as K
import keras.layers as layers
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2_modules import PointnetSAModuleVotes
import pointnet2_utils

mode = "--stable"

def atenke_dikodirowanya(net, fin_line, num_class, nhb, nsc, mean):
    
    batch_size = net.transpose(2,1).shape[0]
    prop_num = net.transpose(2,1).shape[1]

    fin_line['o_score'] = net.transpose(2,1)[:,:,0:2]
    
    center = fin_line['aggregated_vote_xyz'] + net.transpose(2,1)[:,:,2:5] 
    fin_line['center'] = center

    nhbs = 5+nhb

    h_score = net.transpose(2,1)[:,:,5:nhbs]
    h_res_norm = net.transpose(2,1)[:,:,nhbs:nhbs*2]

    fin_line['h_score'] = h_score
     
    fin_line['h_res_norm'] = h_res_norm
    fin_line['h_res'] = h_res_norm * (np.pi/nhb)

    nhbnsc = nhbs*2+nsc

    size_scores = net.transpose(2,1)[:,:,nhbs*2:nhbnsc]
    s_res_norm = net.transpose(2,1)[:,:,nhbnsc:nhbnsc*4].view([batch_size, prop_num, nsc, 3])
    fin_line['size_scores'] = size_scores
    fin_line['s_res_norm'] = s_res_norm
    pool_m = mean.astype(np.float32)
    fin_line['size_residuals'] = s_res_norm * torch.as_tensor(pool_m).cuda().unsqueeze(0).unsqueeze(0)

    cls_score = net.transpose(2,1)[:,:,nhbnsc*4:]

    if cls_score >= 0:
        fin_line['sem_cls_scores'] = cls_score

    return fin_line


class ProposalModule(nn.Module):
    def __init__(self, num_class, nhb, nsc, mean, prop_num, sampling, seed_feat_dim=256):
        super().__init__() 

        self.sampling = sampling
        self.mean = mean

        self.batch1 = torch.nn.BatchNorm1d(128)
        self.batch2 = torch.nn.BatchNorm1d(128)

        self.nsc = nsc

        self.c1 = torch.nn.Conv1d(128,128,1)
        self.c2 = torch.nn.Conv1d(128,128,1)

        self.num_class = num_class
        self.prop_num = prop_num

        self.seed_feat_dim = seed_feat_dim
        self.nhb = nhb


        self.c3 = torch.nn.Conv1d(128,2+3+nhb*2+nsc*4+self.num_class,1)


        self.vote_aggregation = PointnetSAModuleVotes( 
            npoint=self.prop_num,
            radius=0.5,
            nsample=16,
            mlp=[self.seed_feat_dim, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True
        )
    


    def forward(self, xyz, features, fin_line):

        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds

        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(fin_line['seed_xyz'], self.prop_num)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)

        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = fin_line['seed_xyz'].shape[1]
            batch_size = fin_line['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.prop_num), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)

        else:
            exit()

        fin_line['aggregated_vote_xyz'] = xyz # (batch_size, prop_num, 3)
        fin_line['aggregated_vote_inds'] = sample_inds # (batch_size, prop_num,) # should be 0,1,2,...,prop_num

        # --------- PROPOSAL GENERATION ---------
        net = F.relu(self.batch1(self.c1(features))) 
        net = F.relu(self.batch2(self.c2(net)))

        # New 'functionality' 
        net = self.c3(net) # (batch_size, 2+3+nhb*2+nsc*4, prop_num)

        fin_line = atenke_dikodirowanya(net, fin_line, self.num_class, self.nhb, self.nsc, self.mean)
        return fin_line

if __name__=='__main__':
    rand1 = torch.rand(8,1024,3)
    rand2 = torch.rand(8,256,1024)
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))

    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC

    net = ProposalModule(DC.num_class, DC.nhb,
        DC.nsc, DC.mean,
        128, 'seed_fps').cuda()

    fin_line = {'seed_xyz': rand1.cuda()}
    output = net(rand1.cuda(), rand2.cuda(), fin_line)
    for val in output:
        print("The key: ", val, output[val].shape)

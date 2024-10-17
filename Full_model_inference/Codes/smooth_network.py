import torch
import torch.nn as nn
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform
import utils.torch_tps_transform_point as torch_tps_transform_point
import ssl
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.models as models
import torchvision.transforms as T
import random
from torch import nn, einsum
from einops import rearrange

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W



def build_SmoothNet(net, tsmotion_list1, tsmotion_list2, smesh_list1, smesh_list2):

    # predict the delta motion for tsflow
    ori_mesh1, ori_mesh2, ori_path1, ori_path2, delta_motion1, delta_motion2 = net(smesh_list1, smesh_list2, tsmotion_list1, tsmotion_list2)


    # get the smoothes tsflow
    smooth_path1 = ori_path1 + delta_motion1
    smooth_path2 = ori_path2 + delta_motion2
    # get the actual warping mesh
    smooth_mesh1 = ori_mesh1 - delta_motion1  # bs, T, h, w, 2
    smooth_mesh2 = ori_mesh2 - delta_motion2  # bs, T, h, w, 2



    out_dict = {}
    out_dict.update(ori_path1 = ori_path1, smooth_path1 = smooth_path1, ori_mesh1 = ori_mesh1, smooth_mesh1 = smooth_mesh1, ori_path2 = ori_path2, smooth_path2 = smooth_path2, ori_mesh2 = ori_mesh2, smooth_mesh2 = smooth_mesh2)

    return out_dict




# define and forward
class SmoothNet(nn.Module):

    def __init__(self, dropout=0.):
        super(SmoothNet, self).__init__()


        self.MotionPre = MotionPrediction()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    # forward
    def forward(self, smesh_list1, smesh_list2, tsmotion_list1, tsmotion_list2):

        tsflow_list1 = [tsmotion_list1[0]]
        for i in range(1, len(tsmotion_list1)):
            tsflow_list1.append(tsflow_list1[i-1] + tsmotion_list1[i])

        tsflow_list2 = [tsmotion_list2[0]]
        for i in range(1, len(tsmotion_list2)):
            tsflow_list2.append(tsflow_list2[i-1] + tsmotion_list2[i])


        smesh1 = torch.cat(smesh_list1, 3)  # bs, h, w, 2*T
        smesh2 = torch.cat(smesh_list2, 3)  # bs, h, w, 2*T
        tsflow1 = torch.cat(tsflow_list1, 3)  # bs, h, w, 2*T
        tsflow2 = torch.cat(tsflow_list2, 3)  # bs, h, w, 2*T

        # reshape
        smesh1 = smesh1.reshape(-1, grid_h+1, grid_w+1, len(smesh_list1), 2)     # bs, h, w, T, 2
        smesh1 = smesh1.permute(0, 3, 1, 2, 4)   # bs, T, h, w, 2

        smesh2 = smesh2.reshape(-1, grid_h+1, grid_w+1, len(smesh_list2), 2)     # bs, h, w, T, 2
        smesh2 = smesh2.permute(0, 3, 1, 2, 4)   # bs, T, h, w, 2

        tsflow1 = tsflow1.reshape(-1, grid_h+1, grid_w+1, len(tsflow_list1), 2)     # bs, h, w, T, 2
        tsflow1 = tsflow1.permute(0, 3, 1, 2, 4)   # bs, T, h, w, 2

        tsflow2 = tsflow2.reshape(-1, grid_h+1, grid_w+1, len(tsflow_list2), 2)     # bs, h, w, T, 2
        tsflow2 = tsflow2.permute(0, 3, 1, 2, 4)   # bs, T, h, w, 2



        delta_tsflow = self.MotionPre(smesh1, smesh2, tsflow1, tsflow2)
        delta_tsflow1 = delta_tsflow[...,0:2]
        delta_tsflow2 = delta_tsflow[...,2:4]



        return smesh1, smesh2, tsflow1, tsflow2, delta_tsflow1, delta_tsflow2




class MotionPrediction(nn.Module):
    def __init__(self, kernel = 5):
        super().__init__()

        self.embedding1 = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        self.embedding2 = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU()
        )
        self.embedding3 = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )

        self.pad = kernel // 2
        self.MotionConv3D = nn.Sequential(
            nn.Conv3d(128, 128, (kernel, 3, 3), padding=(self.pad, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(128, 128, (kernel, 3, 3), padding=(self.pad, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(128, 128, (kernel, 3, 3), padding=(self.pad, 1, 1)),
            nn.ReLU()
        )

        self.decoding = nn.Sequential(
            nn.Linear(128, 4)
        )



    def forward(self, smesh1, smesh2, tsflow1, tsflow2):

        # predict the delta motion to stabilize the last meshflow
        hidden11 = self.embedding1(smesh1)       # bs, T, H, W, 32
        # hidden12 = self.embedding2(omask1)       # bs, T, H, W, 32
        hidden13 = self.embedding3(tsflow1)       # bs, T, H, W, 32
        hidden1 = torch.cat([hidden11, hidden13], 4)    # bs, T, H, W, 64

        hidden21 = self.embedding1(smesh2)       # bs, T, H, W, 32
        # hidden22 = self.embedding2(omask2)       # bs, T, H, W, 32
        hidden23 = self.embedding3(tsflow2)       # bs, T, H, W, 32
        hidden2 = torch.cat([hidden21, hidden23], 4)    # bs, T, H, W, 64

        hidden = torch.cat([hidden1, hidden2], 4) # bs, T, H, W, 128

        hidden = self.MotionConv3D(hidden.permute(0, 4, 1, 2, 3))   # bs, 128, T, H, W
        delta_tsflow = self.decoding(hidden.permute(0, 2, 3, 4, 1))   # bs, T, H, W, 4

        return delta_tsflow




import torch
import torch.nn as nn
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform
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




def build_TemporalNet(net, img_tensor_list):
    batch_size, _, img_h, img_w = img_tensor_list[0].size()
    frame_num = len(img_tensor_list)

    motion_list = net(img_tensor_list)
    motion_list.insert(0, torch.zeros([batch_size, grid_h+1, grid_w+1,2]).cuda())

    out_dict = {}
    out_dict.update(motion_list = motion_list)


    return out_dict




def get_res18_FeatureMap(resnet18_model):


    layers_list = []


    layers_list.append(resnet18_model.conv1)    #stride 2*2     H/2
    layers_list.append(resnet18_model.bn1)
    layers_list.append(resnet18_model.relu)
    layers_list.append(resnet18_model.maxpool)  #stride 2       H/4

    layers_list.append(resnet18_model.layer1)                  #H/4
    layers_list.append(resnet18_model.layer2)                  #H/8

    feature_extractor_stage1 = nn.Sequential(*layers_list)
    feature_extractor_stage2 = nn.Sequential(resnet18_model.layer3)


    return feature_extractor_stage1, feature_extractor_stage2

# define and forward
class TemporalNet(nn.Module):

    def __init__(self, dropout=0.):
        super(TemporalNet, self).__init__()

        self.regressNet2_part1 = nn.Sequential(
            nn.Conv2d(49, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 23, 40

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 12, 20

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 6, 10

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
            # 3, 5
        )

        self.regressNet2_part2 = nn.Sequential(
            nn.Linear(in_features=1536, out_features=1024, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=(grid_w+1)*(grid_h+1)*2, bias=True)

        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        resnet18_model = models.resnet.resnet18(weights="DEFAULT")
        if torch.cuda.is_available():
            resnet18_model = resnet18_model.cuda()
        self.feature_extractor_stage1, self.feature_extractor_stage2 = get_res18_FeatureMap(resnet18_model)


    # forward
    def forward(self, img_tensor_list):
        batch_size, _, img_h, img_w = img_tensor_list[0].size()
        frame_num = len(img_tensor_list)

        Mesh_motion_list = []

        #---------------------------------------------------------------------
        feature1 = 0
        feature2 = 0
        for i in range(0, frame_num-1):
            if i == 0 :
                feature1 = self.feature_extractor_stage1(img_tensor_list[0].cuda())
                feature2 = self.feature_extractor_stage1(img_tensor_list[1].cuda())
            else:
                feature2 = self.feature_extractor_stage1(img_tensor_list[i+1].cuda())

            # cost volume and regression
            cv2 = self.cost_volume(feature1, feature2, search_range=3, norm=False)
            temp_2 = self.regressNet2_part1(cv2)
            temp_2 = temp_2.view(temp_2.size()[0], -1)
            offset_2 = self.regressNet2_part2(temp_2)
            M_motion_2 = offset_2.reshape(-1, grid_h+1, grid_w+1, 2)

            Mesh_motion_list.append(M_motion_2)

            feature1 = feature2.clone()

        return Mesh_motion_list

    @staticmethod
    def cost_volume(x1, x2, search_range, norm=True, fast=True):
        if norm:
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = F.normalize(x2, p=2, dim=1)
        bs, c, h, w = x1.shape
        padded_x2 = F.pad(x2, [search_range] * 4)  # [b,c,h,w] -> [b,c,h+sr*2,w+sr*2]
        max_offset = search_range * 2 + 1

        if fast:
            # faster(*2) but cost higher(*n) GPU memory
            patches = F.unfold(padded_x2, (max_offset, max_offset)).reshape(bs, c, max_offset ** 2, h, w)
            cost_vol = (x1.unsqueeze(2) * patches).mean(dim=1, keepdim=False)
        else:
            # slower but save memory
            cost_vol = []
            for j in range(0, max_offset):
                for i in range(0, max_offset):
                    x2_slice = padded_x2[:, :, j:j + h, i:i + w]
                    cost = torch.mean(x1 * x2_slice, dim=1, keepdim=True)
                    cost_vol.append(cost)
            cost_vol = torch.cat(cost_vol, dim=1)

        cost_vol = F.leaky_relu(cost_vol, 0.1)

        return cost_vol

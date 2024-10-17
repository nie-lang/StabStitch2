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


#Covert global homo into mesh
def H2Mesh(H, rigid_mesh):

    H_inv = torch.inverse(H)
    ori_pt = rigid_mesh.reshape(rigid_mesh.size()[0], -1, 2)
    ones = torch.ones(rigid_mesh.size()[0], (grid_h+1)*(grid_w+1),1)
    if torch.cuda.is_available():
        ori_pt = ori_pt.cuda()
        ones = ones.cuda()

    ori_pt = torch.cat((ori_pt, ones), 2) # bs*(grid_h+1)*(grid_w+1)*3
    tar_pt = torch.matmul(H_inv, ori_pt.permute(0,2,1)) # bs*3*(grid_h+1)*(grid_w+1)

    mesh_x = torch.unsqueeze(tar_pt[:,0,:]/tar_pt[:,2,:], 2)
    mesh_y = torch.unsqueeze(tar_pt[:,1,:]/tar_pt[:,2,:], 2)
    mesh = torch.cat((mesh_x, mesh_y), 2).reshape([rigid_mesh.size()[0], grid_h+1, grid_w+1, 2])

    return mesh


def get_rigid_mesh(batch_size, height, width):


    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt

def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) # bs*(grid_h+1)*(grid_w+1)*2

    return norm_mesh.reshape([batch_size, -1, 2]) # bs*-1*2

def recover_mesh(norm_mesh, height, width):
    #from [bs, pn, 2] to [bs, grid_h+1, grid_w+1, 2]

    batch_size = norm_mesh.size()[0]
    mesh_w = (norm_mesh[...,0]+1) * float(width) / 2.
    mesh_h = (norm_mesh[...,1]+1) * float(height) / 2.
    mesh = torch.stack([mesh_w, mesh_h], 2) # [bs,(grid_h+1)*(grid_w+1),2]

    return mesh.reshape([batch_size, grid_h+1, grid_w+1, 2])




def build_model(net, tmotion_tensor_list1, tmotion_tensor_list2, smotion_tensor_list1, smotion_tensor_list2, img_list1, img_list2):

    batch_size, _, img_h, img_w = img_list1[0].size()


    ##############################################
    #############   data preparation  ############
    # converting tmotion (t-th frame) into tsmotion ( (t-1)-th frame )
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    smesh_list1 = []
    tsmotion_list1 = []
    smesh_list2 = []
    tsmotion_list2 = []
    for i in range(len(tmotion_tensor_list1)):
        smotion1 = smotion_tensor_list1[i]
        smesh1 = rigid_mesh + smotion1
        smotion2 = smotion_tensor_list2[i]
        smesh2 = rigid_mesh + smotion2
        if i == 0:
            tsmotion1 = smotion1.clone() * 0   # define the initial position as zero
            tsmotion2 = smotion2.clone() * 0   # define the initial position as zero
        else:
            smotion1_1 = smotion_tensor_list1[i-1]
            smesh1_1 = rigid_mesh + smotion1_1
            tmotion1 = tmotion_tensor_list1[i]
            tmesh1 = rigid_mesh + tmotion1
            norm_smesh1_1 = get_norm_mesh(smesh1_1, img_h, img_w)
            norm_tmesh1 = get_norm_mesh(tmesh1, img_h, img_w)
            tsmesh1 = torch_tps_transform_point.transformer(norm_tmesh1, norm_rigid_mesh, norm_smesh1_1)
            tsmotion1 = recover_mesh(tsmesh1, img_h, img_w) - smesh1

            smotion2_1 = smotion_tensor_list2[i-1]
            smesh2_1 = rigid_mesh + smotion2_1
            tmotion2 = tmotion_tensor_list2[i]
            tmesh2 = rigid_mesh + tmotion2
            norm_smesh2_1 = get_norm_mesh(smesh2_1, img_h, img_w)
            norm_tmesh2 = get_norm_mesh(tmesh2, img_h, img_w)
            tsmesh2 = torch_tps_transform_point.transformer(norm_tmesh2, norm_rigid_mesh, norm_smesh2_1)
            tsmotion2 = recover_mesh(tsmesh2, img_h, img_w) - smesh2


        smesh_list1.append(smesh1)
        tsmotion_list1.append(tsmotion1)
        smesh_list2.append(smesh2)
        tsmotion_list2.append(tsmotion2)

    # predict the delta motion for tsflow
    stitch_mesh1, stitch_mesh2, ori_path1, ori_path2, delta_motion1, delta_motion2 = net(smesh_list1, smesh_list2, tsmotion_list1, tsmotion_list2)

    # get the smoothes tsflow
    smooth_path1 = ori_path1 + delta_motion1
    smooth_path2 = ori_path2 + delta_motion2

    # get the actual warping mesh
    target_mesh1 = stitch_mesh1 - delta_motion1  # bs, T, h, w, 2
    target_mesh2 = stitch_mesh2 - delta_motion2  # bs, T, h, w, 2


    ################################################################################
    ################################################################################
    ## prepare the data for trajectory consistency term and online alignment term ##
    out_dict = {}

    out_dict.update(tsmotion_list1 = tsmotion_list1, ori_path1 = ori_path1, smooth_path1 = smooth_path1, ori_mesh1 =stitch_mesh1, target_mesh1 = target_mesh1)
    out_dict.update(tsmotion_list2 = tsmotion_list2, ori_path2 = ori_path2, smooth_path2 = smooth_path2, ori_mesh2 =stitch_mesh2, target_mesh2 = target_mesh2)
    # colaboration of two views
    # get imgs (Note: we only get the last images to reduce the training GPU memory)
    img1 = img_list1[-1] # bs, 3, h, w
    img2 = img_list2[-1] # bs, 3, h, w
    mask_img = torch.ones_like(img1[:,0,...].unsqueeze(1)).cuda()
    norm_target_mesh1 = get_norm_mesh(target_mesh1[:,-1,...], img_h, img_w)
    norm_target_mesh2 = get_norm_mesh(target_mesh2[:,-1,...], img_h, img_w)
    rigid_mesh = get_rigid_mesh(img1.shape[0], img_h, img_w)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    out1_img = torch_tps_transform.transformer(torch.cat([img1, mask_img], 1), norm_target_mesh1, norm_rigid_mesh,(img_h, img_w))
    out2_img = torch_tps_transform.transformer(torch.cat([img2, mask_img], 1), norm_target_mesh2, norm_rigid_mesh,(img_h, img_w))
    ovmask_img = (out1_img[:,-1,...] * out2_img[:,-1,...]).unsqueeze(1)
    out_dict.update(wimg1 = out1_img[:,0:3,...], wimg2 = out2_img[:,0:3,...], ovmask_img = ovmask_img)
    ####################### attention: align_corners=True ##############
    # step 1: resize smooth path to the image resolution  # from [bs, T, grid_h+1, grid_w+1, 2]  to  [bs * T, 2, img_h,img_w]
    # in fact, we only resize them to 1/4 resolution for reducing GPU memory
    dense_spath1 = smooth_path1.reshape(-1, grid_h+1, grid_w+1, 2) # bs*t, grid_h+1, grid_w+1, 2
    dense_spath1 = dense_spath1.permute(0, 3, 1, 2)  # bs*t, 2, grid_h+1, grid_w+1
    dense_spath1 = F.interpolate(dense_spath1, (int(img_h/4), int(img_w/4)), mode='bicubic', align_corners=True)  #bs*t, 2, img_h, img_w
    dense_spath2 = smooth_path2.reshape(-1, grid_h+1, grid_w+1, 2) # bs*t, grid_h+1, grid_w+1, 2
    dense_spath2 = dense_spath2.permute(0, 3, 1, 2)  # bs*t, 2, grid_h+1, grid_w+1
    dense_spath2 = F.interpolate(dense_spath2, (int(img_h/4), int(img_w/4)), mode='bicubic', align_corners=True)  #bs*t, 2, img_h,
    #######################
    # step 2: get overlapping masks from target mesh12
    mask_spath = torch.ones_like(dense_spath1[:,0,...].unsqueeze(1)).cuda()
    rigid_mesh = get_rigid_mesh(dense_spath1.shape[0], img_h, img_w)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_target_mesh1 = get_norm_mesh(target_mesh1.reshape(-1, grid_h+1, grid_w+1, 2), img_h, img_w)
    norm_target_mesh2 = get_norm_mesh(target_mesh2.reshape(-1, grid_h+1, grid_w+1, 2), img_h, img_w)
    out1_spath = torch_tps_transform.transformer(torch.cat([dense_spath1, mask_spath], 1), norm_target_mesh1,norm_rigid_mesh, (int(img_h/4), int(img_w/4)))
    out2_spath = torch_tps_transform.transformer(torch.cat([dense_spath2, mask_spath], 1), norm_target_mesh2,norm_rigid_mesh, (int(img_h/4), int(img_w/4)))
    ovmask_spath = (out1_spath[:,-1,...] * out2_spath[:,-1,...]).unsqueeze(1)
    # calculate loss for consistency overlapping path
    out_dict.update(dense_spath1 = out1_spath[:,0:2,...], dense_spath2 = out2_spath[:,0:2,...], ovmask_spath =ovmask_spath)


    return out_dict



# define and forward
class Network(nn.Module):

    def __init__(self, dropout=0.):
        super(Network, self).__init__()


        self.MotionPre = MotionPrediction()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    # forward
    def forward(self, smesh_list1, smesh_list2, tsmotion_list1, tsmotion_list2):

        # to generate meshflow from the first motion
        tsflow_list1 = [tsmotion_list1[0]]
        for i in range(1, len(tsmotion_list1)):
            tsflow_list1.append(tsflow_list1[i-1] + tsmotion_list1[i])

        tsflow_list2 = [tsmotion_list2[0]]
        for i in range(1, len(tsmotion_list2)):
            tsflow_list2.append(tsflow_list2[i-1] + tsmotion_list2[i])

        # generate meshflow tensor
        smesh1 = torch.cat(smesh_list1, 3)  # bs, h, w, 2*T
        smesh2 = torch.cat(smesh_list2, 3)  # bs, h, w, 2*T
        # omask1 = torch.stack(omask_list1, 3)  # bs, h, w, T
        # omask2 = torch.stack(omask_list2, 3)  # bs, h, w, T
        tsflow1 = torch.cat(tsflow_list1, 3)  # bs, h, w, 2*T
        tsflow2 = torch.cat(tsflow_list2, 3)  # bs, h, w, 2*T

        # reshape
        smesh1 = smesh1.reshape(-1, grid_h+1, grid_w+1, len(smesh_list1), 2)     # bs, h, w, T, 2
        smesh1 = smesh1.permute(0, 3, 1, 2, 4)   # bs, T, h, w, 2

        smesh2 = smesh2.reshape(-1, grid_h+1, grid_w+1, len(smesh_list2), 2)     # bs, h, w, T, 2
        smesh2 = smesh2.permute(0, 3, 1, 2, 4)   # bs, T, h, w, 2

        # omask1 = omask1.reshape(-1, grid_h+1, grid_w+1, len(omask_list1), 1)     # bs, h, w, T, 1
        # omask1 = omask1.permute(0, 3, 1, 2, 4)   # bs, T, h, w, 2

        # omask2 = omask2.reshape(-1, grid_h+1, grid_w+1, len(omask_list2), 1)     # bs, h, w, T, 1
        # omask2 = omask2.permute(0, 3, 1, 2, 4)   # bs, T, h, w, 2

        tsflow1 = tsflow1.reshape(-1, grid_h+1, grid_w+1, len(tsflow_list1), 2)     # bs, h, w, T, 2
        tsflow1 = tsflow1.permute(0, 3, 1, 2, 4)   # bs, T, h, w, 2

        tsflow2 = tsflow2.reshape(-1, grid_h+1, grid_w+1, len(tsflow_list2), 2)     # bs, h, w, T, 2
        tsflow2 = tsflow2.permute(0, 3, 1, 2, 4)   # bs, T, h, w, 2



        delta_tsflow = self.MotionPre(smesh1, smesh2, tsflow1, tsflow2)
        delta_tsflow1 = delta_tsflow[...,0:2]
        delta_tsflow2 = delta_tsflow[...,2:4]



        return smesh1, smesh2, tsflow1, tsflow2, delta_tsflow1, delta_tsflow2




# input: bs, 2, grid_h, grid_w (list, len=3)
# output: bs, 2, grid_h, grid_w
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
        # input: meshflow -- bs, T, h, w, 2
        # output: delta_meshflow -- bs, T, h, w, 2

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




# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from spatial_network import build_SpatialNet, SpatialNet
from temporal_network import build_TemporalNet, TemporalNet
from smooth_network import build_SmoothNet, SmoothNet
import os
import numpy as np
import skimage
import cv2
import utils.torch_tps_transform as torch_tps_transform
import utils.torch_tps_transform_point as torch_tps_transform_point
from PIL import Image
import glob
import time
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'full_model_ssd')



def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2)**l_num))

#target_mesh: bs, T, h, w, 2
def inter_grid_loss(mesh):

    ##############################
    # compute horizontal edges
    w_edges = mesh[:,:,:,0:grid_w,:] - mesh[:,:,:,1:grid_w+1,:]
    # compute angles of two successive horizontal edges
    cos_w = torch.sum(w_edges[:,:,:,0:grid_w-1,:] * w_edges[:,:,:,1:grid_w,:],3) / (torch.sqrt(torch.sum(w_edges[:,:,:,0:grid_w-1,:]*w_edges[:,:,:,0:grid_w-1,:],3))*torch.sqrt(torch.sum(w_edges[:,:,:,1:grid_w,:]*w_edges[:,:,:,1:grid_w,:],3)))
    # horizontal angle-preserving error for two successive horizontal edges
    delta_w_angle = 1 - cos_w
    # horizontal angle-preserving error for two successive horizontal grids
    delta_w_angle = delta_w_angle[:,:,0:grid_h,:] + delta_w_angle[:,:,1:grid_h+1,:]
    ##############################

    ##############################
    # compute vertical edges
    h_edges = mesh[:,:,0:grid_h,:,:] - mesh[:,:,1:grid_h+1,:,:]
    # compute angles of two successive vertical edges
    cos_h = torch.sum(h_edges[:,:,0:grid_h-1,:,:] * h_edges[:,:,1:grid_h,:,:],3) / (torch.sqrt(torch.sum(h_edges[:,:,0:grid_h-1,:,:]*h_edges[:,:,0:grid_h-1,:,:],3))*torch.sqrt(torch.sum(h_edges[:,:,1:grid_h,:,:]*h_edges[:,:,1:grid_h,:,:],3)))
    # vertical angle-preserving error for two successive vertical edges
    delta_h_angle = 1 - cos_h
    # vertical angle-preserving error for two successive vertical grids
    delta_h_angle = delta_h_angle[:,:,:,0:grid_w] + delta_h_angle[:,:,:,1:grid_w+1]
    ##############################

    # successive depth grid on the horizontal dimension
    error_w = delta_w_angle
    # successive depth grid on the vertical dimension
    error_h = delta_h_angle

    return torch.mean(error_w) + torch.mean(error_h)



# intra-grid constraint
def intra_grid_loss(pts):

    max_w = 480/grid_w * 2
    max_h = 360/grid_h * 2

    delta_x = pts[:,:,:,1:grid_w+1,0] - pts[:,:,:,0:grid_w,0]
    delta_y = pts[:,:,1:grid_h+1,:,1] - pts[:,:,0:grid_h,:,1]

    loss_x = F.relu(delta_x - max_w)
    loss_y = F.relu(delta_y - max_h)
    loss = torch.mean(loss_x) + torch.mean(loss_y)

    #loss = loss_x[:,0:grid_h,:] + loss_x[:,1:grid_h+1,:] + loss_y[:,:,0:grid_w] + loss_y[:,:,1:grid_w+1]
    #loss = torch.mean(loss*overlap)

    return loss

def linear_blender(ref, tgt, ref_m, tgt_m, mask=False):
    blur = GaussianBlur(kernel_size=(21,21), sigma=20)
    r1, c1 = torch.nonzero(ref_m[0, 0], as_tuple=True)
    r2, c2 = torch.nonzero(tgt_m[0, 0], as_tuple=True)

    center1 = (r1.float().mean(), c1.float().mean())
    center2 = (r2.float().mean(), c2.float().mean())

    vec = (center2[0] - center1[0], center2[1] - center1[1])

    ovl = (ref_m * tgt_m).round()[:, 0].unsqueeze(1)
    ref_m_ = ref_m[:, 0].unsqueeze(1) - ovl
    r, c = torch.nonzero(ovl[0, 0], as_tuple=True)

    ovl_mask = torch.zeros_like(ref_m_).cuda()
    proj_val = (r - center1[0]) * vec[0] + (c - center1[1]) * vec[1]
    ovl_mask[ovl.bool()] = (proj_val - proj_val.min()) / (proj_val.max() - proj_val.min() + 1e-3)

    mask1 = (blur(ref_m_ + (1-ovl_mask)*ref_m[:,0].unsqueeze(1)) * ref_m + ref_m_).clamp(0,1)
    if mask: return mask1

    mask2 = (1-mask1) * tgt_m
    stit = ref * mask1 + tgt * mask2

    return stit


def recover_mesh(norm_mesh, height, width):
    #from [bs, pn, 2] to [bs, grid_h+1, grid_w+1, 2]

    batch_size = norm_mesh.size()[0]
    mesh_w = (norm_mesh[...,0]+1) * float(width) / 2.
    mesh_h = (norm_mesh[...,1]+1) * float(height) / 2.
    mesh = torch.stack([mesh_w, mesh_h], 2) # [bs,(grid_h+1)*(grid_w+1),2]

    return mesh.reshape([batch_size, grid_h+1, grid_w+1, 2])

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



# bs, T, h, w, 2  smooth_path
def get_stable_sqe(img1_list, img2_list, smooth_mesh1, smooth_mesh2):
    batch_size, _, img_h, img_w = img2_list[0].shape
    print(img2_list[0].shape)

    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)


    stable_list1 = []
    stable_list2 = []
    # mesh_tran_list = []
    for i in range(len(img2_list)):

        mesh1 = smooth_mesh1[:,i,:,:,:]
        norm_mesh1 = get_norm_mesh(mesh1, img_h, img_w)
        img1 = (img1_list[i].cuda()+1)*127.5

        mesh2 = smooth_mesh2[:,i,:,:,:]
        norm_mesh2 = get_norm_mesh(mesh2, img_h, img_w)
        img2 = (img2_list[i].cuda()+1)*127.5

        mask = torch.ones_like(img2).cuda()
        img1_warp = torch_tps_transform.transformer(torch.cat([img1, mask], 1), norm_mesh1, norm_rigid_mesh, (img_h, img_w), mode = 'NORMAL')
        img2_warp = torch_tps_transform.transformer(torch.cat([img2, mask], 1), norm_mesh2, norm_rigid_mesh, (img_h, img_w), mode = 'NORMAL')

        stable_list1.append(img1_warp[0].cpu().detach().numpy().transpose(1,2,0))
        stable_list2.append(img2_warp[0].cpu().detach().numpy().transpose(1,2,0))



    return stable_list1, stable_list2



def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # define the network
    spatial_net = SpatialNet()
    temporal_net = TemporalNet()
    smooth_net = SmoothNet()
    if torch.cuda.is_available():
        spatial_net = spatial_net.cuda()
        temporal_net = temporal_net.cuda()
        smooth_net = smooth_net.cuda()

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) == 3:
        # load spatial warp model
        spatial_model_path = MODEL_DIR + "/spatial_warp.pth"
        spatial_checkpoint = torch.load(spatial_model_path)
        spatial_net.load_state_dict(spatial_checkpoint['model'])
        print('load model from {}!'.format(spatial_model_path))
        # load temporal warp model
        temporal_model_path = MODEL_DIR + "/temporal_warp.pth"
        temporal_checkpoint = torch.load(temporal_model_path)
        temporal_net.load_state_dict(temporal_checkpoint['model'])
        print('load model from {}!'.format(temporal_model_path))
        # load smooth warp model
        smooth_model_path = MODEL_DIR + "/smooth_warp.pth"
        smooth_checkpoint = torch.load(smooth_model_path)
        smooth_net.load_state_dict(smooth_checkpoint['model'])
        print('load model from {}!'.format(smooth_model_path))
    else:
        print('No checkpoint found!')
        exit(0)


    spatial_net.eval()
    temporal_net.eval()
    smooth_net.eval()

    print("##################start testing#######################")


    video_name_list = glob.glob(os.path.join(args.test_path, '*'))
    video_name_list = sorted(video_name_list)
    print(video_name_list)

    RE_video_name = ["00000107", "00000101", "MR002", "S13", "S28"]
    LL_video_name = ["0000074", "0000085", "0000090", "0000099", "00000100"]
    LT_video_name = ["0000021", "0000037", "0000040", "00000140", "ML001"]
    MF_video_name = ["00000168", "00000175", "00000224", "MR006", "SF34"]
    RE_stability_list = []
    RE_distortion_list = []
    LL_stability_list = []
    LL_distortion_list = []
    LT_stability_list = []
    LT_distortion_list = []
    MF_stability_list = []
    MF_distortion_list = []
    stability_list = []
    distortion_list = []

    RE_psnr_list = []
    RE_ssim_list = []
    LL_psnr_list = []
    LL_ssim_list = []
    LT_psnr_list = []
    LT_ssim_list = []
    MF_psnr_list = []
    MF_ssim_list = []
    psnr_list = []
    ssim_list = []


    count = 0


    for i in range(len(video_name_list)):

        print()
        print(i)
        print(video_name_list[i])

        #define an online buffer (len == 7)
        buffer_len = 7
        tmotion_tensor_list1 = []
        smotion_tensor_list1 = []
        tmotion_tensor_list2 = []
        smotion_tensor_list2 = []

        # img name list
        img1_name_list = glob.glob(os.path.join(video_name_list[i]+ "/video1/", '*.jpg'))
        img2_name_list = glob.glob(os.path.join(video_name_list[i]+ "/video2/", '*.jpg'))
        img1_name_list = sorted(img1_name_list)
        img2_name_list = sorted(img2_name_list)


        #img1_list = []
        img1_tensor_list = []
        img2_tensor_list = []

        img_h = 360
        img_w = 480
        # load imgs
        for k in range(0, len(img2_name_list)):
            img1 = cv2.imread(img1_name_list[k])
            img1 = cv2.resize(img1, (img_w, img_h))
            img1 = img1.astype(dtype=np.float32)
            img1 = np.transpose(img1, [2, 0, 1])
            img1 = (img1 / 127.5) - 1.0
            img1_tensor = torch.tensor(img1).unsqueeze(0)
            img1_tensor_list.append(img1_tensor)

            img2 = cv2.imread(img2_name_list[k])
            img2 = cv2.resize(img2, (img_w, img_h))
            img2 = img2.astype(dtype=np.float32)
            img2 = np.transpose(img2, [2, 0, 1])
            img2 = (img2 / 127.5) - 1.0
            img2_tensor = torch.tensor(img2).unsqueeze(0)
            img2_tensor_list.append(img2_tensor)


        start_time1 = time.time()
        NOF = len(img2_name_list)
        # motion estimation
        for k in range(0, len(img2_name_list)):

            # step 1: spatial warp
            with torch.no_grad():
                spatial_batch_out = build_SpatialNet(spatial_net, img1_tensor_list[k].cuda(), img2_tensor_list[k].cuda())
            smotion1 = spatial_batch_out['motion1']
            smotion2 = spatial_batch_out['motion2']
            smotion_tensor_list1.append(smotion1)
            smotion_tensor_list2.append(smotion2)

        # step 2: temporal warp
        with torch.no_grad():
            temporal_batch_out1 = build_TemporalNet(temporal_net, img1_tensor_list)
            temporal_batch_out2 = build_TemporalNet(temporal_net, img2_tensor_list)
        tmotion_tensor_list1 = temporal_batch_out1['motion_list']
        tmotion_tensor_list2 = temporal_batch_out2['motion_list']


        print("fps (spatial & temporal warp):")
        print(NOF/(time.time() - start_time1))


        ##############################################
        #############   data preparation  ############
        # converting tmotion (t-th frame) into tsmotion ( (t-1)-th frame )
        rigid_mesh = get_rigid_mesh(1, img_h, img_w)
        norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
        smesh_list1 = []
        smesh_list2 = []
        tsmotion_list1 = []
        tsmotion_list2 = []
        for k in range(len(tmotion_tensor_list1)):
            smotion1 = smotion_tensor_list1[k]
            smesh1 = rigid_mesh + smotion1
            smotion2 = smotion_tensor_list2[k]
            smesh2 = rigid_mesh + smotion2
            if k == 0:
                tsmotion1 = smotion1.clone() * 0
                tsmotion2 = smotion2.clone() * 0
            else:
                smotion1_1 = smotion_tensor_list1[k-1]
                smesh1_1 = rigid_mesh + smotion1_1
                tmotion1 = tmotion_tensor_list1[k]
                tmesh1 = rigid_mesh + tmotion1
                norm_smesh1_1 = get_norm_mesh(smesh1_1, img_h, img_w)
                norm_tmesh1 = get_norm_mesh(tmesh1, img_h, img_w)
                tsmesh1 = torch_tps_transform_point.transformer(norm_tmesh1, norm_rigid_mesh, norm_smesh1_1)
                tsmotion1 = recover_mesh(tsmesh1, img_h, img_w) - smesh1

                smotion2_1 = smotion_tensor_list2[k-1]
                smesh2_1 = rigid_mesh + smotion2_1
                tmotion2 = tmotion_tensor_list2[k]
                tmesh2 = rigid_mesh + tmotion2
                norm_smesh2_1 = get_norm_mesh(smesh2_1, img_h, img_w)
                norm_tmesh2 = get_norm_mesh(tmesh2, img_h, img_w)
                tsmesh2 = torch_tps_transform_point.transformer(norm_tmesh2, norm_rigid_mesh, norm_smesh2_1)
                tsmotion2 = recover_mesh(tsmesh2, img_h, img_w) - smesh2


            # append
            smesh_list1.append(smesh1)
            smesh_list2.append(smesh2)
            tsmotion_list1.append(tsmotion1)
            tsmotion_list2.append(tsmotion2)


        # step 3: smooth warp
        ori_mesh1 = 0
        smooth_mesh1 = 0
        delta_motion1 = 0

        ori_mesh2 = 0
        smooth_mesh2 = 0
        delta_motion2 = 0

        # ori_path
        ori_path2 = 0
        smooth_path2 = 0


        for k in range(len(tmotion_tensor_list1)-6):

            tsmotion_sublist1 = tsmotion_list1[k:k+7]
            tsmotion_sublist1[0] = tsmotion_sublist1[0] * 0

            tsmotion_sublist2 = tsmotion_list2[k:k+7]
            tsmotion_sublist2[0] = tsmotion_sublist2[0] * 0


            with torch.no_grad():
                smooth_batch_out = build_SmoothNet(smooth_net, tsmotion_sublist1, tsmotion_sublist2, smesh_list1[k:k+7], smesh_list2[k:k+7])

            _ori_mesh1 = smooth_batch_out["ori_mesh1"]
            _smooth_mesh1 = smooth_batch_out["smooth_mesh1"]

            _ori_mesh2 = smooth_batch_out["ori_mesh2"]
            _smooth_mesh2 = smooth_batch_out["smooth_mesh2"]

            _ori_path2 = smooth_batch_out["ori_path2"]
            _smooth_path2 = smooth_batch_out["smooth_path2"]


            if k == 0:
                ori_mesh1 = _ori_mesh1
                smooth_mesh1 = _smooth_mesh1

                ori_mesh2 = _ori_mesh2
                smooth_mesh2 = _smooth_mesh2

                ori_path2 = _ori_path2
                smooth_path2 = _smooth_path2

            else:
                # for ref
                ori_mesh1 = torch.cat((ori_mesh1, _ori_mesh1[:,-1,...].unsqueeze(1)), 1)
                smooth_mesh1 = torch.cat((smooth_mesh1, _smooth_mesh1[:,-1,...].unsqueeze(1)), 1)

                # for tgt
                ori_mesh2 = torch.cat((ori_mesh2, _ori_mesh2[:,-1,...].unsqueeze(1)), 1)
                smooth_mesh2 = torch.cat((smooth_mesh2, _smooth_mesh2[:,-1,...].unsqueeze(1)), 1)

                new_ori_path2 = ori_path2[:,-1,...] + (_ori_path2[:,-1,...] - _ori_path2[:,-2,...])
                ori_path2 = torch.cat((ori_path2, new_ori_path2.unsqueeze(1)), 1)
                new_smooth_path2 = ori_path2[:,-1,...] + (_smooth_path2[:,-1,...] - _ori_path2[:,-1,...])
                smooth_path2 = torch.cat((smooth_path2, new_smooth_path2.unsqueeze(1)), 1)


        print("fps (smooth warp):")
        print(NOF/(time.time() - start_time1))

        ##########################################################

        print("smoothness score: original and smooth")
        # calculate smooth score (original path)
        smooth_path_lll = ori_path2[:,:-6,:,:,:]
        smooth_path_ll = ori_path2[:,1:-5,:,:,:]
        smooth_path_l = ori_path2[:,2:-4,:,:,:]
        smooth_path_mid = ori_path2[:,3:-3,:,:,:]
        smooth_path_r = ori_path2[:,4:-2,:,:,:]
        smooth_path_rr = ori_path2[:,5:-1,:,:,:]
        smooth_path_rrr = ori_path2[:,6:,:,:,:]
        smooth_loss_ori = (l_num_loss(smooth_path_lll, smooth_path_mid, 2)+l_num_loss(smooth_path_rrr, smooth_path_mid, 2))* 0.1
        smooth_loss_ori += (l_num_loss(smooth_path_ll, smooth_path_mid, 2)+l_num_loss(smooth_path_rr, smooth_path_mid, 2)) * 0.3
        smooth_loss_ori += (l_num_loss(smooth_path_l, smooth_path_mid, 2)+l_num_loss(smooth_path_r, smooth_path_mid, 2)) * 0.9
        print(smooth_loss_ori)

        # calculate smooth score (smooth path)
        smooth_path_lll = smooth_path2[:,:-6,:,:,:]
        smooth_path_ll = smooth_path2[:,1:-5,:,:,:]
        smooth_path_l = smooth_path2[:,2:-4,:,:,:]
        smooth_path_mid = smooth_path2[:,3:-3,:,:,:]
        smooth_path_r = smooth_path2[:,4:-2,:,:,:]
        smooth_path_rr = smooth_path2[:,5:-1,:,:,:]
        smooth_path_rrr = smooth_path2[:,6:,:,:,:]
        smooth_loss = (l_num_loss(smooth_path_lll, smooth_path_mid, 2)+l_num_loss(smooth_path_rrr, smooth_path_mid, 2))* 0.1
        smooth_loss += (l_num_loss(smooth_path_ll, smooth_path_mid, 2)+l_num_loss(smooth_path_rr, smooth_path_mid, 2)) * 0.3
        smooth_loss += (l_num_loss(smooth_path_l, smooth_path_mid, 2)+l_num_loss(smooth_path_r, smooth_path_mid, 2)) * 0.9
        print(smooth_loss)
        #print("---------------")

        #########################################################
        print("distortion score: original and smooth")
        ori_distortion_score = []
        tar_distortion_score = []
        for k in range(ori_mesh2.shape[1]):
            ori_spatial_loss = 1*inter_grid_loss(ori_mesh2[:,k,...].unsqueeze(1)) + 1*intra_grid_loss(ori_mesh2[:,k,...].unsqueeze(1))
            tar_spatial_loss = 1*inter_grid_loss(smooth_mesh2[:,k,...].unsqueeze(1)) + 1*intra_grid_loss(smooth_mesh2[:,k,...].unsqueeze(1))
            ori_distortion_score.append(ori_spatial_loss.item())
            tar_distortion_score.append(tar_spatial_loss.item())
        print(max(ori_distortion_score))
        print(max(tar_distortion_score))
        print("---------------")

        vn = video_name_list[i].split('/')[-1]
        stab_score = smooth_loss.item()
        dist_score = max(tar_distortion_score)
        stability_list.append(stab_score)
        distortion_list.append(dist_score)
        if vn in RE_video_name:
            RE_stability_list.append(stab_score)
            RE_distortion_list.append(dist_score)
        elif vn in LL_video_name:
            LL_stability_list.append(stab_score)
            LL_distortion_list.append(dist_score)
        elif vn in LT_video_name:
            LT_stability_list.append(stab_score)
            LT_distortion_list.append(dist_score)
        elif vn in MF_video_name:
            MF_stability_list.append(stab_score)
            MF_distortion_list.append(dist_score)

        ####################################################################


        stable_list1, stable_list2 = get_stable_sqe(img1_tensor_list, img2_tensor_list, smooth_mesh1, smooth_mesh2)


        print("fps (warping & average blending):")
        print(NOF/(time.time() - start_time1))


        for k in range(len(stable_list1)):

            # calculate PSNR/SSIM
            _img1_warp = stable_list1[k][...,0:3]
            _img1_warp_mask = stable_list1[k][...,3:6]

            _img2_warp = stable_list2[k][...,0:3]
            _img2_warp_mask = stable_list2[k][...,3:6]

            _ovmask = _img1_warp_mask * _img2_warp_mask
            psnr = skimage.measure.compare_psnr(_img1_warp*_ovmask, _img2_warp*_ovmask, 255)
            ssim = skimage.measure.compare_ssim(_img1_warp*_ovmask, _img2_warp*_ovmask, data_range=255, multichannel=True)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

            if vn in RE_video_name:
                RE_psnr_list.append(psnr)
                RE_ssim_list.append(ssim)
            elif vn in LL_video_name:
                LL_psnr_list.append(psnr)
                LL_ssim_list.append(ssim)
            elif vn in LT_video_name:
                LT_psnr_list.append(psnr)
                LT_ssim_list.append(ssim)
            elif vn in MF_video_name:
                MF_psnr_list.append(psnr)
                MF_ssim_list.append(ssim)

            print('i = {}, psnr = {:.6f}'.format( k+1, psnr))

        print('over')

    # show quantitative results
    print("=================== Analysis ==================")
    print("PSNR/SSIM")
    print('RE psnr:', np.mean(RE_psnr_list))
    print('RE ssim:', np.mean(RE_ssim_list))
    print('LL psnr:', np.mean(LL_psnr_list))
    print('LL ssim:', np.mean(LL_ssim_list))
    print('LT psnr:', np.mean(LT_psnr_list))
    print('LT ssim:', np.mean(LT_ssim_list))
    print('MF psnr:', np.mean(MF_psnr_list))
    print('MF ssim:', np.mean(MF_ssim_list))
    print()
    print('average psnr:', np.mean(psnr_list))
    print('average ssim:', np.mean(ssim_list))

    print()
    print("------------------")
    print("stability/distortion")
    print('RE stability:', np.mean(RE_stability_list))
    print('RE distortion:', np.mean(RE_distortion_list))
    print('LL stability:', np.mean(LL_stability_list))
    print('LL distortion:', np.mean(LL_distortion_list))
    print('LT stability:', np.mean(LT_stability_list))
    print('LT distortion:', np.mean(LT_distortion_list))
    print('MF stability:', np.mean(MF_stability_list))
    print('MF distortion:', np.mean(MF_distortion_list))
    print()
    print('average stability:', np.mean(stability_list))
    print('average distortion:', np.mean(distortion_list))
    print("##################end testing#######################")




if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--test_path', type=str, default='/opt/data/private/nl/Data/StabStitch-D/testing/')



    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)

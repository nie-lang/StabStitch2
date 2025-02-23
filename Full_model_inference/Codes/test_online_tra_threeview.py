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

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False

last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'full_model_tra')



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

    warp12_mesh1 = 0.
    warp12_mesh2 = 0.
    warp23_mesh1 = 0.
    warp23_mesh2 = 0.

    img1_list = []
    img2_list = []
    img3_list = []

    video_path1 = args.video1_path
    video_path2 = args.video2_path
    video_path3 = args.video3_path


    for i in range(2):

        if i == 0:
            video_frame_path1 = video_path1
            video_frame_path2 = video_path2
        else:
            video_frame_path1 = video_path2
            video_frame_path2 = video_path3


        #define an online buffer (len == 7)
        buffer_len = 7
        tmotion_tensor_list1 = []
        smotion_tensor_list1 = []
        tmotion_tensor_list2 = []
        smotion_tensor_list2 = []

        # img name list
        img1_name_list = glob.glob(os.path.join(video_frame_path1, '*.jpg'))
        img2_name_list = glob.glob(os.path.join(video_frame_path2, '*.jpg'))
        img1_name_list = sorted(img1_name_list)
        img2_name_list = sorted(img2_name_list)

        img1_tensor_list = []
        img2_tensor_list = []
        img1_hr_tensor_list = []
        img2_hr_tensor_list = []

        img_h = 360
        img_w = 480
        # load imgs
        for k in range(0, len(img2_name_list)):
            img1 = cv2.imread(img1_name_list[k])
            # get high-resolution input
            img1_hr = img1.astype(dtype=np.float32)
            img1_hr = np.transpose(img1_hr, [2, 0, 1])
            img1_hr_tensor = torch.tensor(img1_hr).unsqueeze(0)
            img1_hr_tensor_list.append(img1_hr_tensor)
            # get 360x480 input
            img1 = cv2.resize(img1, (img_w, img_h))
            img1 = img1.astype(dtype=np.float32)
            img1 = np.transpose(img1, [2, 0, 1])
            img1 = (img1 / 127.5) - 1.0
            img1_tensor = torch.tensor(img1).unsqueeze(0)
            img1_tensor_list.append(img1_tensor)

            img2 = cv2.imread(img2_name_list[k])
            # get high-resolution input
            img2_hr = img2.astype(dtype=np.float32)
            img2_hr = np.transpose(img2_hr, [2, 0, 1])
            img2_hr_tensor = torch.tensor(img2_hr).unsqueeze(0)
            img2_hr_tensor_list.append(img2_hr_tensor)
            # get 360x480 input
            img2 = cv2.resize(img2, (img_w, img_h))
            img2 = img2.astype(dtype=np.float32)
            img2 = np.transpose(img2, [2, 0, 1])
            img2 = (img2 / 127.5) - 1.0
            img2_tensor = torch.tensor(img2).unsqueeze(0)
            img2_tensor_list.append(img2_tensor)


        if i == 0:
            img1_list = img1_hr_tensor_list
            img2_list = img2_hr_tensor_list
        else:
            img3_list = img2_hr_tensor_list

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

        for k in range(len(tmotion_tensor_list1)-6):
            #
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


            if k == 0:
                ori_mesh1 = _ori_mesh1
                smooth_mesh1 = _smooth_mesh1

                ori_mesh2 = _ori_mesh2
                smooth_mesh2 = _smooth_mesh2

            else:
                # for ref
                ori_mesh1 = torch.cat((ori_mesh1, _ori_mesh1[:,-1,...].unsqueeze(1)), 1)
                smooth_mesh1 = torch.cat((smooth_mesh1, _smooth_mesh1[:,-1,...].unsqueeze(1)), 1)

                # for tgt
                ori_mesh2 = torch.cat((ori_mesh2, _ori_mesh2[:,-1,...].unsqueeze(1)), 1)
                smooth_mesh2 = torch.cat((smooth_mesh2, _smooth_mesh2[:,-1,...].unsqueeze(1)), 1)


        print("fps (smooth warp):")
        print(NOF/(time.time() - start_time1))

        # get meshes
        if i == 0:
            warp12_mesh1 = smooth_mesh1     # 1, N, grid_h+1, grid_w+1, 2
            warp12_mesh2 = smooth_mesh2
        else:
            warp23_mesh1 = smooth_mesh1
            warp23_mesh2 = smooth_mesh2

    ########################################################################################
    # resize the mesh to the original resolution
    batch_size, _, img_h, img_w = img1_list[0].shape
    warp12_mesh1 = torch.stack([warp12_mesh1[...,0]*img_w/480, warp12_mesh1[...,1]*img_h/360], 4)
    warp12_mesh2 = torch.stack([warp12_mesh2[...,0]*img_w/480, warp12_mesh2[...,1]*img_h/360], 4)
    warp23_mesh1 = torch.stack([warp23_mesh1[...,0]*img_w/480, warp23_mesh1[...,1]*img_h/360], 4)
    warp23_mesh2 = torch.stack([warp23_mesh2[...,0]*img_w/480, warp23_mesh2[...,1]*img_h/360], 4)

    # mesh alignment (we suppose warp12_mesh2 and warp23_mesh1 correspond to the same view)
    offset = (warp12_mesh2 - warp23_mesh1).reshape(warp12_mesh2.shape[0], warp12_mesh2.shape[1], -1, 2)  # bs, N, -1, 2
    offset = torch.mean(offset, 2)  # bs, N, 2
    offset = offset.unsqueeze(2).unsqueeze(2) # bs, N, 1, 1, 2

    # transform mesh coordinate
    warp23_mesh1 = warp23_mesh1 + offset
    warp23_mesh2 = warp23_mesh2 + offset

    # find the middle mesh plane
    middle_mesh = (warp12_mesh2 + warp23_mesh1)/2.

    # predefined canvas
    width_max1 = torch.max(warp12_mesh1[...,0])
    width_max2 = torch.max(warp12_mesh2[...,0])
    width_max3 = torch.max(warp23_mesh1[...,0])
    width_max4 = torch.max(warp23_mesh2[...,0])
    width_max = torch.maximum(width_max1, width_max2)
    width_max = torch.maximum(width_max, width_max3)
    width_max = torch.maximum(width_max, width_max4)
    width_min1 = torch.min(warp12_mesh1[...,0])
    width_min2 = torch.min(warp12_mesh2[...,0])
    width_min3 = torch.min(warp23_mesh1[...,0])
    width_min4 = torch.min(warp23_mesh2[...,0])
    width_min = torch.minimum(width_min1, width_min2)
    width_min = torch.minimum(width_min, width_min3)
    width_min = torch.minimum(width_min, width_min4)
    height_max1 = torch.max(warp12_mesh1[...,1])
    height_max2 = torch.max(warp12_mesh2[...,1])
    height_max3 = torch.max(warp23_mesh1[...,1])
    height_max4 = torch.max(warp23_mesh2[...,1])
    height_max = torch.maximum(height_max1, height_max2)
    height_max = torch.maximum(height_max, height_max3)
    height_max = torch.maximum(height_max, height_max4)
    height_min1 = torch.min(warp12_mesh1[...,1])
    height_min2 = torch.min(warp12_mesh2[...,1])
    height_min3 = torch.min(warp23_mesh1[...,1])
    height_min4 = torch.min(warp23_mesh2[...,1])
    height_min = torch.minimum(height_min1, height_min2)
    height_min = torch.minimum(height_min, height_min3)
    height_min = torch.minimum(height_min, height_min4)

    out_width = width_max - width_min
    out_height = height_max - height_min

    print("predefined canvas")
    print(out_width)
    print(out_height)

    warp12_mesh1 = torch.stack([warp12_mesh1[...,0]-width_min, warp12_mesh1[...,1]-height_min], 4)
    warp12_mesh2 = torch.stack([warp12_mesh2[...,0]-width_min, warp12_mesh2[...,1]-height_min], 4)
    warp23_mesh1 = torch.stack([warp23_mesh1[...,0]-width_min, warp23_mesh1[...,1]-height_min], 4)
    warp23_mesh2 = torch.stack([warp23_mesh2[...,0]-width_min, warp23_mesh2[...,1]-height_min], 4)
    middle_mesh = torch.stack([middle_mesh[...,0]-width_min, middle_mesh[...,1]-height_min], 4)


    warp12_mesh1_framelist = []
    warp23_mesh2_framelist = []
    for i in range(middle_mesh.shape[1]):
        norm_warp12_mesh1 = get_norm_mesh(warp12_mesh1[:,i,...], out_height, out_width)
        norm_warp12_mesh2 = get_norm_mesh(warp12_mesh2[:,i,...], out_height, out_width)
        norm_warp23_mesh1 = get_norm_mesh(warp23_mesh1[:,i,...], out_height, out_width)
        norm_warp23_mesh2 = get_norm_mesh(warp23_mesh2[:,i,...], out_height, out_width)
        norm_middle_mesh = get_norm_mesh(middle_mesh[:,i,...], out_height, out_width)

        norm_warp12_mesh1 = torch_tps_transform_point.transformer(norm_warp12_mesh1, norm_warp12_mesh2, norm_middle_mesh)
        warp12_mesh1_frame = recover_mesh(norm_warp12_mesh1, out_height, out_width)
        norm_warp23_mesh2 = torch_tps_transform_point.transformer(norm_warp23_mesh2, norm_warp23_mesh1, norm_middle_mesh)
        warp23_mesh2_frame = recover_mesh(norm_warp23_mesh2, out_height, out_width)

        warp12_mesh1_framelist.append(warp12_mesh1_frame)
        warp23_mesh2_framelist.append(warp23_mesh2_frame)

    warp12_mesh1 = torch.stack(warp12_mesh1_framelist, 1)
    warp23_mesh2 = torch.stack(warp23_mesh2_framelist, 1)
    print(warp12_mesh1.shape)



    # new canvas
    width_max1 = torch.max(warp12_mesh1[...,0])
    width_max2 = torch.max(middle_mesh[...,0])
    width_max3 = torch.max(warp23_mesh2[...,0])
    width_max = torch.maximum(width_max1, width_max2)
    width_max = torch.maximum(width_max, width_max3)
    width_min1 = torch.min(warp12_mesh1[...,0])
    width_min2 = torch.min(middle_mesh[...,0])
    width_min3 = torch.min(warp23_mesh2[...,0])
    width_min = torch.minimum(width_min1, width_min2)
    width_min = torch.minimum(width_min, width_min3)
    height_max1 = torch.max(warp12_mesh1[...,1])
    height_max2 = torch.max(middle_mesh[...,1])
    height_max3 = torch.max(warp23_mesh2[...,1])
    height_max = torch.maximum(height_max1, height_max2)
    height_max = torch.maximum(height_max, height_max3)
    height_min1 = torch.min(warp12_mesh1[...,1])
    height_min2 = torch.min(middle_mesh[...,1])
    height_min3 = torch.min(warp23_mesh2[...,1])
    height_min = torch.minimum(height_min1, height_min2)
    height_min = torch.minimum(height_min, height_min3)

    out_width = width_max - width_min
    out_height = height_max - height_min

    print("new canvas")
    print(out_width)
    print(out_height)

    batch_size, _, img_h, img_w = img1_list[0].shape
    print(img2_list[0].shape)
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)

    stable_list = []
    print("warping and blending")
    # warp
    for i in range(warp12_mesh1.shape[1]):

        mesh1 = warp12_mesh1[:,i,:,:,:]
        mesh_trans1 = torch.stack([mesh1[...,0]-width_min, mesh1[...,1]-height_min], 3)
        norm_mesh1 = get_norm_mesh(mesh_trans1, out_height, out_width)
        img1 = img1_list[i].cuda()

        mesh2 = middle_mesh[:,i,:,:,:]
        mesh_trans2 = torch.stack([mesh2[...,0]-width_min, mesh2[...,1]-height_min], 3)
        norm_mesh2 = get_norm_mesh(mesh_trans2, out_height, out_width)
        img2 = img2_list[i].cuda()

        mesh3 = warp23_mesh2[:,i,:,:,:]
        mesh_trans3 = torch.stack([mesh3[...,0]-width_min, mesh3[...,1]-height_min], 3)
        norm_mesh3 = get_norm_mesh(mesh_trans3, out_height, out_width)
        img3 = img3_list[i].cuda()

        if args.fusion_mode == 'AVERAGE':
            img_warp = torch_tps_transform.transformer(torch.cat([img1, img2, img3], 0), torch.cat([norm_mesh1, norm_mesh2, norm_mesh3], 0), torch.cat([norm_rigid_mesh, norm_rigid_mesh, norm_rigid_mesh], 0), (out_height.int(), out_width.int()), mode = args.warp_mode)

            img12_fusion = img_warp[0] * (img_warp[0]/ (img_warp[0]+img_warp[1]+1e-6)) + img_warp[1] * (img_warp[1]/ (img_warp[0]+img_warp[1]+1e-6))
            fusion = img12_fusion * (img12_fusion/ (img12_fusion+img_warp[2]+1e-6)) + img_warp[2] * (img_warp[2]/ (img12_fusion+img_warp[2]+1e-6))
        else:
            mask = torch.ones_like(img1[:,0,...].unsqueeze(1)).cuda()
            img1 = torch.cat([img1, mask], 1)
            img2 = torch.cat([img2, mask], 1)
            img3 = torch.cat([img3, mask], 1)
            img_warp = torch_tps_transform.transformer(torch.cat([img1, img2, img3], 0), torch.cat([norm_mesh1, norm_mesh2, norm_mesh3], 0), torch.cat([norm_rigid_mesh, norm_rigid_mesh, norm_rigid_mesh], 0), (out_height.int(), out_width.int()), mode = args.warp_mode)
            mask1 = img_warp[0,3,...].unsqueeze(0).unsqueeze(0)
            mask2 = img_warp[1,3,...].unsqueeze(0).unsqueeze(0)
            mask3 = img_warp[2,3,...].unsqueeze(0).unsqueeze(0)
            img12_fusion = linear_blender(img_warp[0,0:3,...].unsqueeze(0), img_warp[1,0:3,...].unsqueeze(0), mask1, mask2)
            mask12 = mask1 + mask2 - mask1*mask2
            fusion = linear_blender(img12_fusion, img_warp[2,0:3,...].unsqueeze(0), mask12, mask3)
            fusion = fusion[0]

        stable_list.append(fusion.cpu())

    print("begin to write into video")
    # save video
    save_path = '../out.mp4'
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = 30
    media_writer = cv2.VideoWriter(save_path, fourcc, fps, (out_width.int(), out_height.int()))
    for k in range(len(stable_list)):
        ave_fusion = stable_list[k].cpu().numpy().transpose(1,2,0)
        media_writer.write(ave_fusion.astype(np.uint8 ))
    media_writer.release()


    print("##################end testing#######################")


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')

    # the path to load input videos
    # Note: video1 should overlap with video2, and video2 should overlap with video3
    parser.add_argument('--video1_path', type=str, default='/opt/data/private/nl/Data/Tra-Dataset2/case5_2/video1/')
    parser.add_argument('--video2_path', type=str, default='/opt/data/private/nl/Data/Tra-Dataset2/case5_2/video2/')
    parser.add_argument('--video3_path', type=str, default='/opt/data/private/nl/Data/Tra-Dataset2/case5_3/video2/')

    # optional parameter: 'NORMAL' or 'FAST'
    # FAST: use F.grid_sample to interpolate. It's fast, but may produce thin black boundary.
    # NORMAL: use our implemented interpolation function. It's a bit slower, but avoid the black boundary.
    parser.add_argument('--warp_mode', type=str, default='NORMAL') # optional parameter: 'Normal' or 'Fast'
    # optional parameter: 'AVERAGE' or 'LINEAR'
    # AVERAGE: faster but more artifacts
    # LINEAR: slower but less artifacts
    parser.add_argument('--fusion_mode', type=str, default='LINEAR')




    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)

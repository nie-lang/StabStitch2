import torch
import torch.nn as nn
import torch.nn.functional as F

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

def get_vgg19_FeatureMap(vgg_model, input_255, layer_index):

    vgg_mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape((1,3,1,1))
    if torch.cuda.is_available():
        vgg_mean = vgg_mean.cuda()
    vgg_input = input_255-vgg_mean

    x_list = []

    for i in range(0,layer_index+1):
        if i == 0:
            x = vgg_model.features[0](vgg_input)
        else:
            x = vgg_model.features[i](x)
            if i == 6 or i == 13 or i ==24:
                x_list.append(x)

    return x_list

def cal_perception_loss(vgg_model, output_H_ref, output_H_tgt, output_tps_ref, output_tps_tgt):

    overlap = (output_H_ref[:,3,:,:] * output_H_tgt[:,3,:,:]).unsqueeze(1)
    overlap = F.interpolate(overlap, (90, 120), mode='bicubic', align_corners=False)
    H_ref_feature_list = get_vgg19_FeatureMap(vgg_model, (output_H_ref[:,0:3,...]+1)*127.5, 13)
    H_tgt_feature_list = get_vgg19_FeatureMap(vgg_model, (output_H_tgt[:,0:3,...]+1)*127.5, 13)
    H_loss = l_num_loss(H_ref_feature_list[-1]*overlap, H_tgt_feature_list[-1]*overlap, 1) #/ torch.sum(overlap)


    overlap = (output_tps_ref[:,3,:,:] * output_tps_tgt[:,3,:,:]).unsqueeze(1)
    overlap = F.interpolate(overlap, (90, 120), mode='bicubic', align_corners=False)
    TPS_ref_feature_list = get_vgg19_FeatureMap(vgg_model, (output_tps_ref[:,0:3,...]+1)*127.5, 13)
    TPS_tgt_feature_list = get_vgg19_FeatureMap(vgg_model, (output_tps_tgt[:,0:3,...]+1)*127.5, 13)
    TPS_loss = l_num_loss(TPS_ref_feature_list[-1]*overlap, TPS_tgt_feature_list[-1]*overlap, 1)


    loss = H_loss * 3. + TPS_loss * 1.

    return loss

def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2)**l_num))



def cal_lp_loss(output_H_ref, output_H_tgt, output_tps_ref, output_tps_tgt):
    batch_size, _, img_h, img_w = output_H_ref.size()


    overlap = (output_H_ref[:,3,:,:] * output_H_tgt[:,3,:,:]).unsqueeze(1)
    lp_loss_1 = l_num_loss(output_H_ref[:,0:3,:,:]*overlap, output_H_tgt[:,0:3,:,:]*overlap, 1)

    overlap = (output_tps_ref[:,3,:,:] * output_tps_tgt[:,3,:,:]).unsqueeze(1)
    lp_loss_2 = l_num_loss(output_tps_ref[:,0:3,:,:]*overlap, output_tps_tgt[:,0:3,:,:]*overlap, 1)


    lp_loss = 3. * lp_loss_1 + 1. * lp_loss_2

    return lp_loss


def inter_grid_loss(mesh):

    batch_size = mesh.shape[0]

    overlap = torch.ones(batch_size, grid_h, grid_w).cuda()

    ##############################
    # compute horizontal edges
    w_edges = mesh[:,:,0:grid_w,:] - mesh[:,:,1:grid_w+1,:]
    # compute angles of two successive horizontal edges
    cos_w = torch.sum(w_edges[:,:,0:grid_w-1,:] * w_edges[:,:,1:grid_w,:],3) / (torch.sqrt(torch.sum(w_edges[:,:,0:grid_w-1,:]*w_edges[:,:,0:grid_w-1,:],3))*torch.sqrt(torch.sum(w_edges[:,:,1:grid_w,:]*w_edges[:,:,1:grid_w,:],3)))
    # horizontal angle-preserving error for two successive horizontal edges
    delta_w_angle = 1 - cos_w
    # horizontal angle-preserving error for two successive horizontal grids
    delta_w_angle = delta_w_angle[:,0:grid_h,:] + delta_w_angle[:,1:grid_h+1,:]
    ##############################

    ##############################
    # compute vertical edges
    h_edges = mesh[:,0:grid_h,:,:] - mesh[:,1:grid_h+1,:,:]
    # compute angles of two successive vertical edges
    cos_h = torch.sum(h_edges[:,0:grid_h-1,:,:] * h_edges[:,1:grid_h,:,:],3) / (torch.sqrt(torch.sum(h_edges[:,0:grid_h-1,:,:]*h_edges[:,0:grid_h-1,:,:],3))*torch.sqrt(torch.sum(h_edges[:,1:grid_h,:,:]*h_edges[:,1:grid_h,:,:],3)))
    # vertical angle-preserving error for two successive vertical edges
    delta_h_angle = 1 - cos_h
    # vertical angle-preserving error for two successive vertical grids
    delta_h_angle = delta_h_angle[:,:,0:grid_w] + delta_h_angle[:,:,1:grid_w+1]
    ##############################

    # on overlapping regions
    depth_diff_w = (1-torch.abs(overlap[:,:,0:grid_w-1] - overlap[:,:,1:grid_w])) * overlap[:,:,0:grid_w-1]
    error_w = depth_diff_w * delta_w_angle
    # on overlapping regions
    depth_diff_h = (1-torch.abs(overlap[:,0:grid_h-1,:] - overlap[:,1:grid_h,:])) * overlap[:,0:grid_h-1,:]
    error_h = depth_diff_h * delta_h_angle

    return torch.mean(error_w) + torch.mean(error_h)



# intra-grid constraint
def intra_grid_loss(pts):

    max_w = 480/grid_w * 2
    max_h = 360/grid_h * 2

    delta_x = pts[:,:,1:grid_w+1,0] - pts[:,:,0:grid_w,0]
    delta_y = pts[:,1:grid_h+1,:,1] - pts[:,0:grid_h,:,1]

    loss_x = F.relu(delta_x - max_w)
    loss_y = F.relu(delta_y - max_h)
    loss = torch.mean(loss_x) + torch.mean(loss_y)


    return loss



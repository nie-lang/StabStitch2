import torch
import torch.nn as nn
import torch.nn.functional as F

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W



def cal_lp_loss(wimg1, wimg2, overlap):
    batch_size, _, img_h, img_w = wimg1.size()

    delta2 =  torch.sum(wimg1*overlap  -  wimg2*overlap, [2,3])  /  torch.sum(overlap, [2,3])
    wimg2_balance = wimg2 + delta2.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    lp_loss = l_num_loss(wimg1*overlap, wimg2_balance*overlap, 1)

    return lp_loss

def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2)**l_num))



#target_mesh: bs, T, h, w, 2
def inter_grid_loss(mesh):

    #overlap = torch.ones_like(mesh[:,0:grid_h, 0:grid_w,0])

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
    # depth_diff_w = (1-torch.abs(overlap[:,:,0:grid_w-1] - overlap[:,:,1:grid_w])) * overlap[:,:,0:grid_w-1]
    #print(depth_diff_w.size())
    #print(delta_w_angle.size())
    # error_w = depth_diff_w * delta_w_angle
    error_w = delta_w_angle
    # successive depth grid on the vertical dimension
    # depth_diff_h = (1-torch.abs(overlap[:,0:grid_h-1,:] - overlap[:,1:grid_h,:])) * overlap[:,0:grid_h-1,:]
    # error_h = depth_diff_h * delta_h_angle
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

    return loss





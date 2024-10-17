import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import cv2
#from torch_homography_model import build_model
from network import build_model, Network
from datetime import datetime
from dataset import TrainDataset
import glob
from loss import cal_lp_loss, inter_grid_loss, l_num_loss, intra_grid_loss
import torchvision.models as models
import matplotlib.pyplot as plt


# path of project
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
SUMMARY_DIR = os.path.join(last_path, 'summary_tra')
writer = SummaryWriter(log_dir=SUMMARY_DIR)
MODEL_DIR = os.path.join(last_path, 'model_tra')

# create folders if it dose not exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)



def train(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # dataset
    train_data = TrainDataset(data_path=args.train_path, frame_num=args.frame_num+args.train_sqe-1)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True)


    # define the network
    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    # define the optimizer and learning rate
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)

        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        glob_iter = checkpoint['glob_iter']
        scheduler.last_epoch = start_epoch
        print('load model from {}!'.format(model_path))
    else:
        start_epoch = 0
        glob_iter = 0
        print('training from stratch!')



    print("##################start training#######################")
    score_print_fre = 300

    for epoch in range(start_epoch, args.max_epoch):

        print("start epoch {}".format(epoch))
        net.train()
        loss_sigma = 0.

        data_sigma = 0.
        smoothness_sigma = 0.
        shape_sigma = 0.
        trajectory_sigma = 0.
        align_sigma = 0.
        online_sigma = 0.


        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        # training
        for i, batch_value in enumerate(train_loader):

            # print("start")
            tmotion_tensor_list1 = [ tmotion.float().cuda() for tmotion in batch_value[0]]
            tmotion_tensor_list2 = [ tmotion.float().cuda() for tmotion in batch_value[1]]
            smotion_tensor_list1 = [ smotion.float().cuda() for smotion in batch_value[2]]
            smotion_tensor_list2 = [ smotion.float().cuda() for smotion in batch_value[3]]
            img_tensor_list1 = [ img.float().cuda() for img in batch_value[4]]
            img_tensor_list2 = [ img.float().cuda() for img in batch_value[5]]

            # forward, backward, update weights
            optimizer.zero_grad()

            total_loss = 0.
            data_loss = 0.
            smoothness_loss = 0.
            shape_loss = 0.
            trajectory_loss = 0.
            align_loss = 0.
            online_loss = 0.

            # prepare for online collaboration term
            path_diff1 = 0.
            path_diff2 = 0.
            smooth_path_list1 = []
            smooth_path_list2 = []

            for k in range(args.train_sqe):
                batch_out = build_model(net, tmotion_tensor_list1[k:k+7], tmotion_tensor_list2[k:k+7], smotion_tensor_list1[k:k+7], smotion_tensor_list2[k:k+7], img_tensor_list1[k:k+7], img_tensor_list2[k:k+7])

                tsmotion_list1 = batch_out['tsmotion_list1']
                ori_path1 = batch_out['ori_path1']    # bs, T, H, W, 2
                smooth_path1 = batch_out['smooth_path1']  # bs, T, H, W, 2
                target_mesh1 = batch_out['target_mesh1']
                ori_mesh1 = batch_out['ori_mesh1']

                tsmotion_list2 = batch_out['tsmotion_list2']
                ori_path2 = batch_out['ori_path2']    # bs, T, H, W, 2
                smooth_path2 = batch_out['smooth_path2']  # bs, T, H, W, 2
                target_mesh2 = batch_out['target_mesh2']
                ori_mesh2 = batch_out['ori_mesh2']

                ########
                wimg1 = batch_out['wimg1']
                wimg2 = batch_out['wimg2']
                dense_spath1 = batch_out['dense_spath1']
                dense_spath2 = batch_out['dense_spath2']
                ovmask_img = batch_out['ovmask_img']
                ovmask_spath = batch_out['ovmask_spath']


                # for the first window, we calculate the loss for data term, smoothness term, shape preserving term, online alignment term, and trajectory consistency term
                if k == 0:

                    # loss 1: data term
                    data_loss += l_num_loss(ori_path1, smooth_path1, 2)   #
                    data_loss += l_num_loss(ori_path2, smooth_path2, 2)   #

                    # loss 2: smoothness path
                    # for view 1
                    smooth_path1_lll = smooth_path1[:,:-6,:,:,:]
                    smooth_path1_ll = smooth_path1[:,1:-5,:,:,:]
                    smooth_path1_l = smooth_path1[:,2:-4,:,:,:]
                    smooth_path1_mid = smooth_path1[:,3:-3,:,:,:]
                    smooth_path1_r = smooth_path1[:,4:-2,:,:,:]
                    smooth_path1_rr = smooth_path1[:,5:-1,:,:,:]
                    smooth_path1_rrr = smooth_path1[:,6:,:,:,:]
                    smoothness_loss += (l_num_loss(smooth_path1_lll, smooth_path1_mid, 2)+l_num_loss(smooth_path1_rrr, smooth_path1_mid, 2)) * 0.1
                    smoothness_loss += (l_num_loss(smooth_path1_ll, smooth_path1_mid, 2)+l_num_loss(smooth_path1_rr, smooth_path1_mid, 2))   * 0.3
                    smoothness_loss += (l_num_loss(smooth_path1_l, smooth_path1_mid, 2)+l_num_loss(smooth_path1_r, smooth_path1_mid, 2)) *   0.9
                    # for view 2
                    smooth_path2_lll = smooth_path2[:,:-6,:,:,:]
                    smooth_path2_ll = smooth_path2[:,1:-5,:,:,:]
                    smooth_path2_l = smooth_path2[:,2:-4,:,:,:]
                    smooth_path2_mid = smooth_path2[:,3:-3,:,:,:]
                    smooth_path2_r = smooth_path2[:,4:-2,:,:,:]
                    smooth_path2_rr = smooth_path2[:,5:-1,:,:,:]
                    smooth_path2_rrr = smooth_path2[:,6:,:,:,:]
                    smoothness_loss += (l_num_loss(smooth_path2_lll, smooth_path2_mid, 2)+l_num_loss(smooth_path2_rrr, smooth_path2_mid, 2)) * 0.1
                    smoothness_loss += (l_num_loss(smooth_path2_ll, smooth_path2_mid, 2)+l_num_loss(smooth_path2_rr, smooth_path2_mid, 2))   * 0.3
                    smoothness_loss += (l_num_loss(smooth_path2_l, smooth_path2_mid, 2)+l_num_loss(smooth_path2_r, smooth_path2_mid, 2)) *   0.9

                    # loss 3: shape preserving term
                    shape_loss += 1*inter_grid_loss(target_mesh1) + 1*intra_grid_loss(target_mesh1)
                    shape_loss += 1*inter_grid_loss(target_mesh2) + 1*intra_grid_loss(target_mesh2)

                    # loss 4: trajectory consistency term
                    trajectory_loss += l_num_loss(dense_spath1*ovmask_spath, dense_spath2*ovmask_spath, 1)   #L1 version (how about L2?)

                    # loss 5: alignment term
                    align_loss += cal_lp_loss(wimg1, wimg2, ovmask_img)

                # prepare for online collaboration term
                # if we do not add the path_diff, the trajectory from the second window will start from 0, which cannot align with the trajectory from the first window
                if k == 0:
                    smooth_path_list1.append(smooth_path1)
                    smooth_path_list2.append(smooth_path2)
                    # record the diff in the first window
                    path_diff1 = tsmotion_list1[1].unsqueeze(1)
                    path_diff2 = tsmotion_list2[1].unsqueeze(1)
                else:
                    smooth_path_list1.append(smooth_path1 + path_diff1)
                    smooth_path_list2.append(smooth_path2 + path_diff2)

            # loss 6: online collaboration term
            for k in range(args.train_sqe-1):
                online_loss += l_num_loss(smooth_path_list1[k][:,1:,...], smooth_path_list1[k+1][:,:-1,...], 2)   #
                online_loss += l_num_loss(smooth_path_list2[k][:,1:,...], smooth_path_list2[k+1][:,:-1,...], 2)

            # total loss
            total_loss = data_loss*1 + smoothness_loss * 50 + shape_loss * 10 + trajectory_loss * 1 + online_loss * 0.1 + align_loss * 1000

            # gradient back-propagation
            total_loss.backward()
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()

            # print("end")

            # add for summary
            loss_sigma += total_loss.item() # total term
            data_sigma += data_loss.item()
            smoothness_sigma += smoothness_loss.item()
            shape_sigma += shape_loss.item()
            trajectory_sigma += trajectory_loss.item()
            align_sigma += align_loss.item()
            online_sigma += online_loss.item()

            print(glob_iter)
            # print loss etc.
            if i % score_print_fre == 0 and i != 0:
                # average
                average_loss = loss_sigma / score_print_fre
                average_data = data_sigma/ score_print_fre
                average_smoothness = smoothness_sigma/ score_print_fre
                average_shape = shape_sigma / score_print_fre
                average_trajectory = trajectory_sigma/ score_print_fre
                average_align = align_sigma/ score_print_fre
                average_online = online_sigma / score_print_fre

                # set 0
                loss_sigma = 0.
                data_sigma = 0.
                smoothness_sigma = 0.
                shape_sigma = 0.
                trajectory_sigma = 0.
                align_sigma = 0.
                online_sigma = 0.

                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f}  Data loss: {:.4f}  Smoothness loss: {:.4f}  lr={:.8f}".format(epoch + 1, args.max_epoch, i + 1, len(train_loader), average_loss, average_data, average_smoothness, optimizer.state_dict()['param_groups'][0]['lr']))

                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                # loss tensorboard
                writer.add_scalar('total loss', average_loss, glob_iter)
                writer.add_scalar('data_loss', average_data, glob_iter)
                writer.add_scalar('smoothness_loss', average_smoothness, glob_iter)
                writer.add_scalar('trajectory_loss', average_trajectory, glob_iter)
                writer.add_scalar('align_loss', average_align, glob_iter)
                writer.add_scalar('shape_loss', average_shape, glob_iter)
                writer.add_scalar('online_loss', average_online, glob_iter)

            glob_iter += 1

        scheduler.step()
        # save model
        if ((epoch+1) % 10 == 0 or (epoch+1)==args.max_epoch):
            filename ='epoch' + str(epoch+1).zfill(3) + '_model.pth'
            model_save_path = os.path.join(MODEL_DIR, filename)
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter}
            torch.save(state, model_save_path)



if __name__=="__main__":


    print('<==================== setting arguments ===================>\n')

    #nl: create the argument parser
    parser = argparse.ArgumentParser()

    #nl: add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--frame_num', type=int, default=7)
    parser.add_argument('--train_sqe', type=int, default=2)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--train_path', type=str, default='../../TraditionalDataset/')

    #nl: parse the arguments
    args = parser.parse_args()
    print(args)

    print('<==================== jump into training function ===================>\n')
    #nl: rain
    train(args)



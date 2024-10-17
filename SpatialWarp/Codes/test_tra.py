# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from network import build_model, Network
from dataset import *
import os
import numpy as np
import skimage
import cv2


last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'model_tra')



def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # dataset
    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)

    # define the network
    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
        print('load model from {}!'.format(model_path))
    else:
        print('No checkpoint found!')
        exit(0)



    print("##################start testing#######################")
    net.eval()
    last_video_name = "last_video"
    for i, batch_value in enumerate(test_loader):

        inpu1_tesnor = batch_value[0].float()
        inpu2_tesnor = batch_value[1].float()
        data_name = batch_value[2][0]

        if torch.cuda.is_available():
            inpu1_tesnor = inpu1_tesnor.cuda()
            inpu2_tesnor = inpu2_tesnor.cuda()

        with torch.no_grad():
            batch_out = build_model(net, inpu1_tesnor, inpu2_tesnor, is_training=False)

        mesh_ref = batch_out['mesh_ref']
        mesh_tgt = batch_out['mesh_tgt']
        mesh_rigid = batch_out['mesh_rigid']

        motion_ref_np = (mesh_ref-mesh_rigid)[0].cpu().detach().numpy()
        motion_tgt_np = (mesh_tgt-mesh_rigid)[0].cpu().detach().numpy()


        video_name = data_name.split('/')[-3]
        if video_name != last_video_name:
            last_video_name = video_name
            new_folder1 = args.test_path + video_name + "/SpatialMotion1/"
            if not os.path.exists(new_folder1):
                os.makedirs(new_folder1)

            new_folder2 = args.test_path + video_name + "/SpatialMotion2/"
            if not os.path.exists(new_folder2):
                os.makedirs(new_folder2)

        np.save(new_folder1 + data_name.split('/')[-1][:-4]+".npy", motion_ref_np)
        np.save(new_folder2 + data_name.split('/')[-1][:-4]+".npy", motion_tgt_np)

        print('i = {}'.format( i+1))


    print("##################end testing#######################")


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_path', type=str, default='../../TraditionalDataset/')

    args = parser.parse_args()
    print(args)
    test(args)


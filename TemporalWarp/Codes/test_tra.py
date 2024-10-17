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
from PIL import Image



last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'model_tra')



def test_sample(net, input1_tesnor, input2_tesnor):
    net.eval()
    with torch.no_grad():
        batch_out = build_model(net, input1_tesnor, input2_tesnor, is_training=False)
    return batch_out


def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # dataset
    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=False)

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
    psnr_list = []
    ssim_list = []
    last_video_name = "last_video"
    net.eval()
    for i, batch_value in enumerate(test_loader):

        inpu1_tesnor = batch_value[0].float()
        inpu2_tesnor = batch_value[1].float()
        inpu3_tesnor = batch_value[2].float()
        inpu4_tesnor = batch_value[3].float()
        data_name = batch_value[4][0]

        if torch.cuda.is_available():
            inpu1_tesnor = inpu1_tesnor.cuda()
            inpu2_tesnor = inpu2_tesnor.cuda()
            inpu3_tesnor = inpu3_tesnor.cuda()
            inpu4_tesnor = inpu4_tesnor.cuda()

        batch_out_ref = test_sample(net, inpu1_tesnor, inpu2_tesnor)
        batch_out_tgt = test_sample(net, inpu3_tesnor, inpu4_tesnor)

        motion_tensor_ref = batch_out_ref['motion']
        motion_tensor_tgt = batch_out_tgt['motion']

        motion_np_ref = motion_tensor_ref[0].cpu().detach().numpy()
        motion_np_tgt = motion_tensor_tgt[0].cpu().detach().numpy()

        video_name = data_name.split('/')[-3]
        if video_name != last_video_name:
            last_video_name = video_name

            new_folder1 = args.test_path + video_name + "/TemporalMotion1/"
            if not os.path.exists(new_folder1):
                os.makedirs(new_folder1)
            new_folder2 = args.test_path + video_name + "/TemporalMotion2/"
            if not os.path.exists(new_folder2):
                os.makedirs(new_folder2)

            file_name1 = new_folder1 + str(int(data_name.split('/')[-1][:-4])-1).zfill(4)+".npy"
            np.save(file_name1, motion_np_ref*0)
            file_name2 = new_folder2 + str(int(data_name.split('/')[-1][:-4])-1).zfill(4)+".npy"
            np.save(file_name2, motion_np_tgt*0)

            print(file_name1)

        np.save(new_folder1 + data_name.split('/')[-1][:-4]+".npy", motion_np_ref)
        np.save(new_folder2 + data_name.split('/')[-1][:-4]+".npy", motion_np_tgt)
        print(i+1)

    print("##################end testing#######################")


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_path', type=str, default='../../TraditionalDataset/')


    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)
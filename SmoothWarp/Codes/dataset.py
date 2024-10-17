from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random


class TrainDataset(Dataset):
    def __init__(self, data_path, frame_num):

        self.width = 480
        self.height = 360
        self.train_path = data_path
        # total frame number and selected frame number(for training, with random interval)
        self.train_frame_num = 12
        self.selected_frame_num = frame_num
        self.datas = OrderedDict()

        self.datas['TMotion1'] = [[] for _ in range(self.train_frame_num)]
        self.datas['TMotion2'] = [[] for _ in range(self.train_frame_num)]
        self.datas['SMotion1'] = [[] for _ in range(self.train_frame_num)]
        self.datas['SMotion2'] = [[] for _ in range(self.train_frame_num)]

        self.datas['img1'] = [[] for _ in range(self.train_frame_num)]
        self.datas['img2'] = [[] for _ in range(self.train_frame_num)]


        video_names = glob.glob(os.path.join(self.train_path, '*'))
        for video_name in sorted(video_names):

            TMotion_list1 = glob.glob(os.path.join(video_name + "/TemporalMotion1/", '*.npy'))
            TMotion_list1 = sorted(TMotion_list1)
            # skip the videos whose frame number is less than train_frame_num
            if len(TMotion_list1) < self.train_frame_num :
                print(len(TMotion_list1))
                continue
            for i in range(self.train_frame_num):
                self.datas['TMotion1'][i].extend(TMotion_list1[i:i+len(TMotion_list1)-self.train_frame_num+1])

            TMotion_list2 = glob.glob(os.path.join(video_name + "/TemporalMotion2/", '*.npy'))
            TMotion_list2 = sorted(TMotion_list2)
            for i in range(self.train_frame_num):
                self.datas['TMotion2'][i].extend(TMotion_list2[i:i+len(TMotion_list2)-self.train_frame_num+1])

            SMotion_list1 = glob.glob(os.path.join(video_name + "/SpatialMotion1/", '*.npy'))
            SMotion_list1 = sorted(SMotion_list1)
            for i in range(self.train_frame_num):
                self.datas['SMotion1'][i].extend(SMotion_list1[i:i+len(SMotion_list1)-self.train_frame_num+1])

            SMotion_list2 = glob.glob(os.path.join(video_name + "/SpatialMotion2/", '*.npy'))
            SMotion_list2 = sorted(SMotion_list2)
            for i in range(self.train_frame_num):
                self.datas['SMotion2'][i].extend(SMotion_list2[i:i+len(SMotion_list2)-self.train_frame_num+1])


            # add imgs
            img_list1 = glob.glob(os.path.join(video_name + "/video1/", '*.jpg'))
            img_list1 = sorted(img_list1)
            for i in range(self.train_frame_num):
                self.datas['img1'][i].extend(img_list1[i:i+len(img_list1)-self.train_frame_num+1])

            img_list2 = glob.glob(os.path.join(video_name + "/video2/", '*.jpg'))
            img_list2 = sorted(img_list2)
            for i in range(self.train_frame_num):
                self.datas['img2'][i].extend(img_list2[i:i+len(img_list2)-self.train_frame_num+1])


        print(len(self.datas['TMotion1'][0]))

    def __getitem__(self, index):

        # data augmentation
        ran = random.sample(range(0, self.train_frame_num, 1), self.selected_frame_num)
        ran = sorted(ran)

        # load motion
        TMotion_tensor_list1 = []
        SMotion_tensor_list1 = []
        TMotion_tensor_list2 = []
        SMotion_tensor_list2 = []
        img_tensor_list1 = []
        img_tensor_list2 = []

        for i in range(0, self.selected_frame_num):

            # load temporal motion
            tmotion1 = np.load(self.datas['TMotion1'][ran[i]][index]) # size: grid_h * grid_w * 2
            tmotion1 = tmotion1.astype(dtype=np.float32)
            tmesh_tensor1 = torch.tensor(tmotion1)
            TMotion_tensor_list1.append(tmesh_tensor1)

            tmotion2 = np.load(self.datas['TMotion2'][ran[i]][index]) # size: grid_h * grid_w * 2
            tmotion2 = tmotion2.astype(dtype=np.float32)
            tmesh_tensor2 = torch.tensor(tmotion2)
            TMotion_tensor_list2.append(tmesh_tensor2)

            # load spatial motion
            smotion1 = np.load(self.datas['SMotion1'][ran[i]][index]) # size: grid_h * grid_w * 2
            smotion1 = smotion1.astype(dtype=np.float32) #+ (np.random.rand()-0.5)*10
            smesh_tensor1 = torch.tensor(smotion1)
            SMotion_tensor_list1.append(smesh_tensor1)

            smotion2 = np.load(self.datas['SMotion2'][ran[i]][index]) # size: grid_h * grid_w * 2
            smotion2 = smotion2.astype(dtype=np.float32) #+ (np.random.rand()-0.5)*10
            smesh_tensor2 = torch.tensor(smotion2)
            SMotion_tensor_list2.append(smesh_tensor2)


            # load imgs
            img1 = cv2.imread(self.datas['img1'][ran[i]][index])
            img1 = cv2.resize(img1, (self.width, self.height)).astype(dtype=np.float32)
            img1 = (img1 / 127.5) - 1.0
            img1 = np.transpose(img1, [2, 0, 1])
            img_tensor1 = torch.tensor(img1)
            img_tensor_list1.append(img_tensor1)

            img2 = cv2.imread(self.datas['img2'][ran[i]][index])
            img2 = cv2.resize(img2, (self.width, self.height)).astype(dtype=np.float32)
            img2 = (img2 / 127.5) - 1.0
            img2 = np.transpose(img2, [2, 0, 1])
            img_tensor2 = torch.tensor(img2)
            img_tensor_list2.append(img_tensor2)



        return (TMotion_tensor_list1, TMotion_tensor_list2, SMotion_tensor_list1, SMotion_tensor_list2, img_tensor_list1, img_tensor_list2)


    def __len__(self):

        return len(self.datas['TMotion1'][0])

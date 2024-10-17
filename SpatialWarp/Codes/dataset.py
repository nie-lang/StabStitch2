from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random


class TrainDataset(Dataset):
    def __init__(self, data_path):

        self.width = 480
        self.height = 360
        self.train_path = data_path
        self.datas = OrderedDict()

        self.datas['video1'] = []
        self.datas['video2'] = []

        video_names = glob.glob(os.path.join(self.train_path, '*'))
        for video_name in sorted(video_names):
            #filtering
            video1_list = glob.glob(os.path.join(video_name+'/video1', '*.jpg'))
            video1_list = sorted(video1_list)[2:]
            self.datas['video1'].extend(video1_list)

            video2_list = glob.glob(os.path.join(video_name+'/video2', '*.jpg'))
            video2_list = sorted(video2_list)[2:]
            self.datas['video2'].extend(video2_list)
        print(len(self.datas['video1']))

    def __getitem__(self, index):

        # load image1
        input1 = cv2.imread(self.datas['video1'][index])
        input1 = cv2.resize(input1, (self.width, self.height))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])

        # load image2
        input2 = cv2.imread(self.datas['video2'][index])
        input2 = cv2.resize(input2, (self.width, self.height))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])

        # convert to tensor
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)

        if_exchange = random.randint(0,1)
        if if_exchange == 0:
            return (input1_tensor, input2_tensor)
        else:
            return (input2_tensor, input1_tensor)

    def __len__(self):

        return len(self.datas['video1'])



class TestDataset(Dataset):
    def __init__(self, data_path):

        self.width = 480
        self.height = 360
        self.test_path = data_path
        self.datas = OrderedDict()

        self.datas['video1'] = []
        self.datas['video2'] = []

        video_names = glob.glob(os.path.join(self.test_path, '*'))
        for video_name in sorted(video_names):
            #filtering
            video1_list = glob.glob(os.path.join(video_name+'/video1', '*.jpg'))
            # video1_list = sorted(video1_list)[2:]
            video1_list = sorted(video1_list)
            self.datas['video1'].extend(video1_list)

            video2_list = glob.glob(os.path.join(video_name+'/video2', '*.jpg'))
            # video2_list = sorted(video2_list)[2:]
            video2_list = sorted(video2_list)
            self.datas['video2'].extend(video2_list)
        print(len(self.datas['video1']))

    def __getitem__(self, index):

        # load image1
        input1 = cv2.imread(self.datas['video1'][index])
        input1 = cv2.resize(input1, (self.width, self.height))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])

        # load image2
        input2 = cv2.imread(self.datas['video2'][index])
        input2 = cv2.resize(input2, (self.width, self.height))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])

        # convert to tensor
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)

        data_name = self.datas['video2'][index]

        return (input1_tensor, input2_tensor, data_name)

    def __len__(self):

        return len(self.datas['video1'])




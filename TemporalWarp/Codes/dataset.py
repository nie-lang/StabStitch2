from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random

# Note: In the training stage, we only use the frames from video2.
class TrainDataset(Dataset):
    def __init__(self, data_path):

        self.width = 480
        self.height = 360
        self.train_path = data_path
        # total frame number and selected frame number(for training, with random interval)
        self.train_frame_num = 4
        self.selected_frame_num = 2

        self.datas = OrderedDict()
        self.datas['video2'] = [[] for _ in range(self.train_frame_num)]


        video_names = glob.glob(os.path.join(self.train_path, '*'))
        for video_name in sorted(video_names):
            # we only use the frames from video2 to train our model
            video1_list = glob.glob(os.path.join(video_name + "/video2/", '*.jpg'))
            video1_list = sorted(video1_list)
            for i in range(self.train_frame_num):
                self.datas['video2'][i].extend(video1_list[i:len(video1_list)-self.train_frame_num+i+1])

        print(len(self.datas['video2'][0]))

    def __getitem__(self, index):

        # data augmentation
        ran = random.sample(range(0, self.train_frame_num), self.selected_frame_num)
        ran = sorted(ran)

        # load images
        input1 = cv2.imread(self.datas['video2'][ran[0]][index])
        input1 = cv2.resize(input1, (self.width, self.height))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])
        input1_tensor = torch.tensor(input1)

        input2 = cv2.imread(self.datas['video2'][ran[1]][index])
        input2 = cv2.resize(input2, (self.width, self.height))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])
        input2_tensor = torch.tensor(input2)

        return (input1_tensor, input2_tensor)

    def __len__(self):
        return len(self.datas['video2'][0])


# When generating temporal warps, we load frames from video1 and video2.
class TestDataset(Dataset):
    def __init__(self, data_path):

        self.width = 480
        self.height = 360
        self.test_path = data_path
        self.test_frame_num = 2
        self.datas = OrderedDict()
        self.datas['video1'] = [[] for _ in range(self.test_frame_num)]
        self.datas['video2'] = [[] for _ in range(self.test_frame_num)]


        # print(self.test_path)
        video_names = glob.glob(os.path.join(self.test_path, '*'))
        # print(video_names)
        for video_name in sorted(video_names):
            #filtering
            video1_list = glob.glob(os.path.join(video_name + "/video1/", '*.jpg'))
            video1_list = sorted(video1_list)
            for i in range(self.test_frame_num):
                self.datas['video1'][i].extend(video1_list[i:len(video1_list)-self.test_frame_num+i+1])

            video2_list = glob.glob(os.path.join(video_name + "/video2/", '*.jpg'))
            video2_list = sorted(video2_list)
            for i in range(self.test_frame_num):
                self.datas['video2'][i].extend(video2_list[i:len(video2_list)-self.test_frame_num+i+1])

        print(len(self.datas['video2'][0]))

    def __getitem__(self, index):

        # get video_name
        data_name = self.datas['video2'][1][index]

        # load images
        input1 = cv2.imread(self.datas['video1'][0][index])
        input1 = cv2.resize(input1, (self.width, self.height)).astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])
        input1_tensor = torch.tensor(input1)

        input2 = cv2.imread(self.datas['video1'][1][index])
        input2 = cv2.resize(input2, (self.width, self.height)).astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])
        input2_tensor = torch.tensor(input2)

        input3 = cv2.imread(self.datas['video2'][0][index])
        input3 = cv2.resize(input3, (self.width, self.height)).astype(dtype=np.float32)
        input3 = (input3 / 127.5) - 1.0
        input3 = np.transpose(input3, [2, 0, 1])
        input3_tensor = torch.tensor(input3)

        input4 = cv2.imread(self.datas['video2'][1][index])
        input4 = cv2.resize(input4, (self.width, self.height)).astype(dtype=np.float32)
        input4 = (input4 / 127.5) - 1.0
        input4 = np.transpose(input4, [2, 0, 1])
        input4_tensor = torch.tensor(input4)




        return (input1_tensor, input2_tensor, input3_tensor, input4_tensor, data_name)


    def __len__(self):

        return len(self.datas['video2'][0])


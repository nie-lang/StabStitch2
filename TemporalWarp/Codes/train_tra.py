import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import cv2
from network import build_model, Network
from datetime import datetime
from dataset import TrainDataset, TestDataset
import glob
from loss import cal_lp_loss, inter_grid_loss, intra_grid_loss
import torchvision.models as models
import skimage


# path of project
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))

# path to save the summary files
SUMMARY_DIR = os.path.join(last_path, 'summary_tra')
writer = SummaryWriter(log_dir=SUMMARY_DIR)

# path to save the model files
MODEL_DIR = os.path.join(last_path, 'model_tra')



# create folders if it dose not exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)



def train_sample(net, inpu1_tesnor, inpu2_tesnor):
    net.train()
    batch_out = build_model(net, inpu1_tesnor, inpu2_tesnor)
    return batch_out



def train(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # dataset
    train_data = TrainDataset(data_path=args.train_path)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)

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
        overlap_loss_sigma = 0.
        nonoverlap_loss_sigma = 0.


        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        # training
        for i, batch_value in enumerate(train_loader):

            input1_tensor = batch_value[0].float()
            input2_tensor = batch_value[1].float()
            if torch.cuda.is_available():
                input1_tensor = input1_tensor.cuda()
                input2_tensor = input2_tensor.cuda()

            # forward, backward, update weights
            optimizer.zero_grad()
            # the training prcess only takes two consecutive frames as input
            batch_out = train_sample(net, input1_tensor, input2_tensor)

            output_mesh = batch_out['output_mesh']
            mesh = batch_out['mesh']

            ##### calculate loss ####
            # overlap
            overlap_loss = cal_lp_loss(input1_tensor, output_mesh)
            # nonoverlap
            nonoverlap_loss = 5*inter_grid_loss(mesh) + 5*intra_grid_loss(mesh)
            # total loss
            total_loss = overlap_loss + nonoverlap_loss

            # gradient back-propagation
            total_loss.backward()
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()

            overlap_loss_sigma += overlap_loss.item()
            nonoverlap_loss_sigma += nonoverlap_loss.item()
            loss_sigma += total_loss.item()


            print(glob_iter)
            # print loss etc.
            if i % score_print_fre == 0 and i != 0:

                average_loss = loss_sigma / score_print_fre
                average_overlap_loss = overlap_loss_sigma/ score_print_fre
                average_nonoverlap_loss = nonoverlap_loss_sigma/ score_print_fre
                loss_sigma = 0.0
                overlap_loss_sigma = 0.
                nonoverlap_loss_sigma = 0.


                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f}  Loss1: {:.4f} lr={:.8f}".format(epoch + 1, args.max_epoch, i + 1, len(train_loader),
                                          average_loss, average_overlap_loss, optimizer.state_dict()['param_groups'][0]['lr']))
                # visualization
                writer.add_image("img_t0", (input1_tensor[0]+1.)/2., glob_iter)
                writer.add_image("img_t1", (input2_tensor[0]+1.)/2., glob_iter)

                writer.add_image("warp_mesh", (output_mesh[0,0:3,:,:]+1.)/2., glob_iter)
                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                writer.add_scalar('total loss', average_loss, glob_iter)
                writer.add_scalar('overlap_loss', average_overlap_loss, glob_iter)
                writer.add_scalar('average_nonoverlap_loss', average_nonoverlap_loss, glob_iter)

            glob_iter += 1

        scheduler.step()

        # save model
        if ((epoch+1) % 20 == 0 or (epoch+1)==args.max_epoch):
            filename ='epoch' + str(epoch+1).zfill(3) + '_model.pth'
            model_save_path = os.path.join(MODEL_DIR, filename)
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter}
            torch.save(state, model_save_path)



if __name__=="__main__":


    print('<==================== setting arguments ===================>\n')

    # create the argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--train_path', type=str, default='../../TraditionalDataset/')

    # parse the arguments
    args = parser.parse_args()
    print(args)

    print('<==================== jump into training function ===================>\n')
    train(args)



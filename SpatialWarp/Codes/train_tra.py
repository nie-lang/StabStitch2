import argparse
import torch
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from network import build_model, Network
from dataset import TrainDataset, TestDataset
import glob
from loss import cal_lp_loss, inter_grid_loss, intra_grid_loss, cal_perception_loss
import skimage
import numpy as np
import torchvision.models as models


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



def train(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # define dataset
    train_data = TrainDataset(data_path=args.train_path)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=4, shuffle=False, drop_last=False)

    # define the network
    net = Network()
    vgg_model = models.vgg19(pretrained=True)
    if torch.cuda.is_available():
        net = net.cuda()
        vgg_model = vgg_model.cuda()

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
    best_ssim = 0.
    current_ssim = 0.

    for epoch in range(start_epoch, args.max_epoch):

        print("start epoch {}".format(epoch))
        net.train()
        loss_sigma = 0.0
        overlap_loss_sigma = 0.
        nonoverlap_loss_sigma = 0.

        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        for i, batch_value in enumerate(train_loader):

            input1_tensor = batch_value[0].float()
            input2_tensor = batch_value[1].float()
            if torch.cuda.is_available():
                input1_tensor = input1_tensor.cuda()
                input2_tensor = input2_tensor.cuda()

            # forward, backward, update weights
            optimizer.zero_grad()

            batch_out = build_model(net, input1_tensor, input2_tensor)
             # result: homo
            output_H_ref = batch_out['output_H_ref']
            output_H_tgt = batch_out['output_H_tgt']
            # result: tps
            output_tps_ref = batch_out['output_tps_ref']
            output_tps_tgt = batch_out['output_tps_tgt']
            mesh_ref = batch_out['mesh_ref']
            mesh_tgt = batch_out['mesh_tgt']

            # Note: Compared with the training process on StabStitch-D dataset, we add the perceptual loss on traditional datasets. Actually, these traditional datasets are challenging, and only applying the photometric loss will lead to non convergence.
            # calculate loss for overlapping regions
            overlap_loss1 = cal_lp_loss(output_H_ref, output_H_tgt, output_tps_ref, output_tps_tgt)
            overlap_loss2 = cal_perception_loss(vgg_model, output_H_ref, output_H_tgt, output_tps_ref, output_tps_tgt)
            overlap_loss = overlap_loss1 * 1. + overlap_loss2 * 0.001

            # calculate loss for non-overlapping regions
            nonoverlap_loss_ref = 5*inter_grid_loss(mesh_ref) + 5*intra_grid_loss(mesh_ref)
            nonoverlap_loss_tgt = 5*inter_grid_loss(mesh_tgt) + 5*intra_grid_loss(mesh_tgt)
            nonoverlap_loss = nonoverlap_loss_ref + nonoverlap_loss_tgt


            # total loss
            total_loss = overlap_loss + nonoverlap_loss
            total_loss.backward()

            # clip the gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()

            overlap_loss_sigma += overlap_loss.item()
            nonoverlap_loss_sigma += nonoverlap_loss.item()
            loss_sigma += total_loss.item()

            print(glob_iter)

            # record loss and images in tensorboard
            if i % score_print_fre == 0 and i != 0:
                average_loss = loss_sigma / score_print_fre
                average_overlap_loss = overlap_loss_sigma/ score_print_fre
                average_nonoverlap_loss = nonoverlap_loss_sigma/ score_print_fre
                loss_sigma = 0.0
                overlap_loss_sigma = 0.
                nonoverlap_loss_sigma = 0.

                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f}  Overlap Loss: {:.4f}  Non-overlap Loss: {:.4f}  lr={:.8f}".format(epoch + 1, args.max_epoch, i + 1, len(train_loader),
                                          average_loss, average_overlap_loss, average_nonoverlap_loss, optimizer.state_dict()['param_groups'][0]['lr']))
                # visualization
                writer.add_image("output_2H", ((output_H_ref[0,0:3,:,:] + output_H_tgt[0,0:3,:,:])/2 + 1.)/2., glob_iter)
                writer.add_image("output_2Mesh", ((output_tps_ref[0,0:3,:,:] + output_tps_tgt[0,0:3,:,:])/2 + 1.)/2., glob_iter)

                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                writer.add_scalar('total loss', average_loss, glob_iter)
                writer.add_scalar('overlap loss', average_overlap_loss, glob_iter)
                writer.add_scalar('nonoverlap loss', average_nonoverlap_loss, glob_iter)

            glob_iter += 1

        scheduler.step()


        # testing
        if (epoch+1)%1 == 0:

            ssim_list = []
            print("----------- starting testing ----------")
            net.eval()
            for i, batch_value in enumerate(test_loader):
                # calculate ssim every 20 samples
                if i%20 == 0:

                    input1_tensor = batch_value[0].float()
                    input2_tensor = batch_value[1].float()

                    if torch.cuda.is_available():
                        input1_tensor = input1_tensor.cuda()
                        input2_tensor = input2_tensor.cuda()


                    with torch.no_grad():
                        batch_out = build_model(net, input1_tensor, input2_tensor, is_training=False)

                    output_tps_ref = batch_out['output_tps_ref']
                    output_tps_tgt = batch_out['output_tps_tgt']

                    # SSIM
                    output_ref = ((output_tps_ref[0,0:3,...]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
                    output_tgt = ((output_tps_tgt[0,0:3,...]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
                    overlap_mask = output_tps_ref[0,3:6,...] * output_tps_tgt[0,3:6,...]
                    overlap_mask = overlap_mask.cpu().detach().numpy().transpose(1,2,0)
                    ssim = skimage.measure.compare_ssim(output_ref*overlap_mask, output_tgt*overlap_mask, data_range=255, multichannel=True)

                    ssim_list.append(ssim)

            writer.add_scalar('SSIM', np.mean(ssim_list), epoch+1)
            current_ssim =  np.mean(ssim_list)

        # save model
        if (current_ssim > best_ssim):
            filename ='epoch_best_model.pth'
            model_save_path = os.path.join(MODEL_DIR, filename)
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter}
            torch.save(state, model_save_path)
            best_ssim = current_ssim



    print("##################end training#######################")



if __name__=="__main__":


    print('<==================== setting arguments ===================>\n')

    # create the argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=8)
    # select the best model within 80 epochs
    parser.add_argument('--max_epoch', type=int, default=80)
    # Note: we do not split the training/testing sets for the traditional datasets. We just validate our effectiveness on the traditional datasets
    parser.add_argument('--train_path', type=str, default='../../TraditionalDataset/')
    parser.add_argument('--test_path', type=str, default='../../TraditionalDataset/')

    # parse the arguments
    args = parser.parse_args()
    print(args)

    # train
    train(args)



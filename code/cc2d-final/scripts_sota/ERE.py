import argparse
import datetime
import os
from pathlib import Path
import time
import shutil
import yaml
import yamlloader
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch.nn import functional as F
from torch.nn import BCELoss
import pickle

import torch.multiprocessing as mp

from models import *
from datasets import select_dataset_ERE
from utils.mylogger import get_mylogger, set_logger_dir
from utils import to_Image, pred_landmarks, visualize, make_dir, heatmap_argmax
from evaluation.eval import Evaluater
from config import get_cfg_defaults, merge_cfg_datasets, merge_cfg_train
from PIL import Image
import numpy as np

import segmentation_models_pytorch as smp

class UNet_Voting(nn.Module):
    def __init__(self, n_channels, n_classes, non_local=False):
        super(UNet_Voting, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True

        self.vgg =  VGG(pretrained=True)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 128, bilinear)
        self.up5 = Up(128, 64, bilinear)


        if non_local:
            self.non_local_5 = RFB_modified(512, 512)
            self.non_local_4 = RFB_modified(256, 256)
            self.non_local_3 = RFB_modified(128, 128)
        self.non_local = non_local
      
        self.final = nn.Conv2d(64, self.n_classes*3, kernel_size=1, padding=0)

    def forward(self, x):
        _, features = self.vgg.features(x, get_features=True)

        if self.non_local:
            features[4] = self.non_local_5(features[4])
        x = self.up1(features[4], features[3])
        if self.non_local: x = self.non_local_4(x)
        x = self.up2(x, features[2])
        if self.non_local: x = self.non_local_3(x)
        x = self.up3(x, features[1])
        x = self.up4(x, features[0])
        x = self.up5(x)

        x = self.final(x)
        
        heatmap = x[:,:self.n_classes,:,:]
        regression_x = x[:,self.n_classes:2*self.n_classes,:,:]
        regression_y = x[:,2*self.n_classes:,:,:]

        return heatmap, regression_y, regression_x

class Unet(nn.Module):
    def __init__(self, in_channels, no_of_landmarks, non_local=False):
        super(Unet, self).__init__()
        self.unet = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_channels=[256, 128, 64, 32, 32],
            in_channels=in_channels,
            classes=no_of_landmarks,
        )
        self.temperatures = nn.Parameter(torch.ones(1, no_of_landmarks, 1, 1), requires_grad=False)

    def forward(self, x):
        return self.unet(x), x, x

    def scale(self, x):
        y = x / self.temperatures
        return y

def two_d_softmax(x):
    exp_y = torch.exp(x)
    return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)


def nll_across_batch(output, target):
    output = torch.clamp(output.double(), min=1e-300, max=1)
    nll = -target * torch.log(output)
    return torch.mean(torch.sum(nll, dim=(2, 3)))


class Tester(object):
    def __init__(self, logger, cfg, runs_dir, args=None):
        
        # # For anthor Testset, deprecated
        # dataset_2 = Cephalometric(config['dataset_pth'], 'Test2')
        # self.dataloader_2 = DataLoader(dataset_2, batch_size=1,
        #                         shuffle=False, num_workers=config['num_workers'])
        self.cfg = cfg
        self.args = args
        self.runs_dir = runs_dir
        # Creat evluater to record results
        self.evaluater = Evaluater(logger, cfg.dataset.num_landmarks, cfg.dataset.eval_radius)

        self.logger = logger
        self.id_landmarks = [i for i in range(cfg.dataset.num_landmarks)]

        self.visual_dir = os.path.join(self.runs_dir, 'visuals')
        make_dir(self.visual_dir)

        self.uncertainty_dir = os.path.join(self.runs_dir, 'uncertainty')
        make_dir(self.uncertainty_dir)  

        self.pred_dir = os.path.join(self.runs_dir, 'pred')
        make_dir(self.pred_dir)  

        self.visual_dir = os.path.join(self.runs_dir, 'visuals')
        make_dir(self.visual_dir)  

        self.dump_dir = os.path.join(self.runs_dir, 'pseudo_labels')
        make_dir(self.dump_dir)  


    def test(self, net, mode="", dump_label=False):
        self.temp_dir = os.path.join(self.runs_dir, 'temp')
        make_dir(self.temp_dir)

        self.evaluater.reset()

        datset_class = select_dataset_ERE(self.cfg.dataset.dataset)
        dataset_1 = datset_class(self.cfg, mode, size=cfg.train.input_size, do_repeat=False)
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                shuffle=False, num_workers=self.cfg.train.num_workers)
        self.Radius = dataset_1.Radius

        img_dict = dict()
        for img, _, __, ___, landmark_list, id_str, scale_rate in tqdm(self.dataloader_1):
            img = img.cuda()
            id_str = id_str[0]

            with torch.no_grad():
                heatmap, regression_y, regression_x = net(img)

            heatmap = two_d_softmax(heatmap)

            vis_heatmap = heatmap[0][3] / heatmap[0][3].max()
            gray_to_PIL(vis_heatmap.cpu().detach())\
                .save(os.path.join(self.visual_dir, id_str+'_heatmap.png'))
            
            pred_landmark = heatmap_argmax(heatmap)
            
            self.evaluater.record(pred_landmark, landmark_list, scale_rate)
            self.evaluater.save_img(img, pred_landmark, landmark_list, self.runs_dir, id_str)

        self.evaluater.cal_metrics()
        self.logger.debug('Calculating metrics Done')

        shutil.rmtree(self.temp_dir)
        self.logger.debug('Remove Trash Done')


def L1Loss(pred, gt, mask=None):
    # L1 Loss for offset map
    assert (pred.shape == gt.shape)
    gap = pred - gt
    distence = gap.abs()
    if mask is not None:
        # Caculate grad of the area under mask
        distence = distence * mask
    return distence.sum() / mask.sum()

def gray_to_PIL(tensor):
    tensor = tensor  * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images

def focal_loss(pred, gt):
    pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
    return (-(1 - pred) * gt * torch.log(pred) - pred * (1 - gt) * torch.log(1 - pred)).mean()


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    alpha = 0
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='gold', help="name of the run")
    parser.add_argument("--dataset", default='head', choices=['head', 'hand', 'leg', 'chest'], help="dataset")
    parser.add_argument("--epoch", type=int, default=0, help="default configs")
    parser.add_argument("--test", type=int, default=0, help="default configs")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    merge_cfg_datasets(cfg, args.dataset)
    merge_cfg_train(cfg, 'ERE')

    # Create Logger
    logger = get_mylogger()
    logger.info(cfg)

    # Create runs dir
    tag = str(datetime.datetime.now()).replace(' ', '-') if args.tag == '' else args.tag
    cfg.tag = tag


    runs_dir = os.path.join("../../final_runs/", tag)
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        runs_path.mkdir()
    cfg.runs_dir = runs_dir
    set_logger_dir(logger, runs_dir)

    # net = UNet(3, config['num_landmarks'])
    network_class = Unet
    # network_class = UNet_Voting
    net = network_class(3, cfg.dataset.num_landmarks, non_local=False)
    net = torch.nn.DataParallel(net)
    net = net.cuda()
    # logger.info(net)

    ema_net = network_class(3, cfg.dataset.num_landmarks, non_local=False)
    ema_net = torch.nn.DataParallel(ema_net)
    ema_net = ema_net.cuda()
    for param in ema_net.parameters():
        param.detach_()



    # Tester
    tester = Tester(logger, cfg, runs_dir=runs_dir)
    # tester.test(net, mode='Test')


    start_epoch = 0
    if args.test:
        logger.info("Loading checkpoints")
        checkpoints = torch.load(os.path.join(cfg.runs_dir, "model_epoch.pth"))
        net.load_state_dict(checkpoints)
        tester = Tester(logger, cfg, runs_dir=runs_dir)
        tester.test(net, mode='Test')
        import sys
        sys.exit(1)

    optimizer = optim.Adam(params=net.parameters(), \
                           lr=cfg.train.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    scheduler = StepLR(optimizer, cfg.train.decay_step, gamma=cfg.train.decay_gamma)

    # loss
    # weight = torch.tensor([1,1,1,1,1,1,1,1]).float().cuda()
    # loss_logic_fn = BCELoss(reduction='none')
    loss_logic_fn = focal_loss
    loss_regression_fn = L1Loss
    torch.autograd.set_detect_anomaly(True)

    if args.test:
        pass

    dataset_class = select_dataset_ERE(args.dataset)
    dataset = dataset_class(cfg, mode='Train', size=cfg.train.input_size)
    dataset.__getitem__(0)
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size,
                drop_last=True, shuffle=True, num_workers=cfg.train.num_workers)

    num_iter = 0
    
    for epoch in range(start_epoch, cfg.train.num_epochs):
        logic_loss_list = list()
        net.train()

        # net.eval()
        # tester.test(net, epoch=epoch, mode="test1", do_voting=True)

        
        for img, mask, offset_y, offset_x, landmark_list, _, __ in tqdm(dataloader):

            img, mask, offset_y, offset_x = img.cuda(), \
                mask.cuda(), offset_y.cuda(), offset_x.cuda()
            
            heatmap, regression_y, regression_x = net(img)

            output = two_d_softmax(heatmap.double())

            logic_loss = nll_across_batch(output, mask)
            loss = logic_loss * cfg.train.loss_lambda
            # loss = (loss * weight).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logic_loss_list.append(loss.cpu().item())

            update_ema_variables(net, ema_net, 0.99, num_iter)
            num_iter += 1
            
        logger.info("Epoch {} Training logic loss {} ". \
                    format(epoch, sum(logic_loss_list) / dataset.__len__()))
        scheduler.step()

        # for img, guassian_mask, landmark_list in tqdm(tester.dataloader_1):
        #     img, guassian_mask = img.cuda(), guassian_mask.cuda()
        #     with torch.no_grad():
        #         heatmap = net(img)
        #     loss = loss_logic_fn(heatmap, guassian_mask)

        #     logic_loss_list.append(loss.cpu().item())
        # logger.info("Epoch {} Testing logic loss {}". \
        #             format(epoch, sum(logic_loss_list) / tester.dataset.__len__()))
        # # save model
        if (epoch + 1) % cfg.train.save_seq == 0:
            # logger.info(runs_dir + "/model_epoch_{}.pth".format(epoch))
            # torch.save(net.state_dict(), runs_dir + "/model_epoch_{}.pth".format(epoch))

            # config['last_epoch'] = epoch
            # net.eval()
            # tester.test(net, epoch=epoch)

            logger.info(runs_dir + "/model_epoch.pth")
            torch.save(net.state_dict(), runs_dir + "/model_epoch.pth")

            cfg.train.last_epoch = epoch
            net.eval()
            tester.test(net, mode='Test')

        # net.eval()
        # tester.test(net, epoch=epoch, train="Train", dump_label=True)

        # dump yaml
        with open(runs_dir + "/config.txt", "w") as f:
            f.write(cfg.dump())

    # Test
    ema_net.eval()
    # import ipdb; ipdb.set_trace()
    tester.test(ema_net, mode='Infer_Train')

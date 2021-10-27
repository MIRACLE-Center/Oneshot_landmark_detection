import argparse
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import BCELoss
import os
import yaml
import yamlloader
import json
from PIL import Image
from pathlib import Path

from models.network import UNet_Pretrained
from datasets.ceph_st import Cephalometric
from utils.eval import Evaluater
from utils.utils_st import to_Image, pred_landmarks, visualize, make_dir, voting

from tutils import tdir, tfilename

def gray_to_PIL(tensor):
    tensor = tensor  * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images


def L1Loss(pred, gt, mask=None):
    assert(pred.shape == gt.shape)
    gap = pred - gt
    distence = gap.abs()
    if mask is not None:
        distence = distence * mask
    return distence.sum() / mask.sum()
    # return distence.mean()

def total_loss(mask, guassian_mask, heatmap, gt_y, gt_x, pred_y, pred_x, lamda, target_list=None):
    b, k, h, w = mask.shape
    logic_loss = BCELoss()
    loss_list = list()
    for i in range(mask.shape[1]):
        channel_loss = 2 * logic_loss(heatmap[0][i], guassian_mask[0][i]) +\
            (L1Loss(pred_y[0][i], gt_y[0][i], mask[0][i]) + L1Loss(pred_x[0][i], gt_x[0][i], mask[0][i]))
        loss_list.append(channel_loss)
    total_loss = np.array(loss_list).sum()
    return total_loss


class Tester(object):
    def __init__(self, logger, config, net=None, tag=None, args=None, test_mode=0):
        dataset_1 = Cephalometric(config['dataset']['pth'], 'Train')
        self.config = config
        self.args = args
        
        self.model = net 
        
        # Creat evluater to record results
        self.test_mode = test_mode
        if test_mode == 0:
            "Cephalometric Test 1"
            self.evaluater = Evaluater(logger, [384, 384],
                                       [2400, 1935])
        elif test_mode == 1:
            "Cephalometric Test 2"
            self.evaluater = Evaluater(logger, [384, 384],
                                       [2400, 1935])
        elif test_mode == 2:
            "Cephalometric Test 1+2"
            self.evaluater = Evaluater(logger, [384, 384],
                                       [2400, 1935])
        else:
            raise ValueError("test_mode should in [1,2,3] but GOT {}".format(test_mode))

        self.logger = logger

        self.dataset = dataset_1
        self.id_landmarks = [i for i in range(config['training']['num_landmarks'])]

    def test(self, net=None, epoch=None, train="", dump_label=False):
        self.evaluater.reset()
        if net is not None:
            self.model = net
        assert(hasattr(self, 'model'))
        ID = 1

        # mode = "Test1" if train == "" else "Train"
        if self.test_mode == 0:
            mode = "Test1"
        elif self.test_mode == 1:
            mode = "Test2"
        elif self.test_mode == 2:
            mode = "Test1+2"
        else: raise ValueError
        dataset_1 = Cephalometric(self.config['dataset']['pth'], mode, do_repeat=False)
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                shuffle=False, num_workers=self.config['training']['num_workers'])
        self.Radius = dataset_1.Radius

        voting_list = [[] for _ in range(19)]
        error_list = [[] for _ in range(19)]
        for img, _, __, ___, landmark_list in tqdm(self.dataloader_1):
            img = img.cuda()

            heatmap, regression_y, regression_x = self.model(img)

            # gray_to_PIL(heatmap[0][1].cpu().detach())\
            #     .save(tfilename('visuals', str(ID)+'_heatmap.png'))
            # Vote for the final accurate point
            pred_landmark, votings = voting(\
                heatmap, regression_y, regression_x, self.Radius, get_voting=True)

            self.evaluater.record(pred_landmark, landmark_list)
            
            # Optional Save viusal results
            # image_pred = visualize(img, pred_landmark, landmark_list)
            # image_pred.save(tfilename('visuals', str(ID)+'_pred.png'))

            if dump_label:
                assert(epoch is not None)
                inference_marks = {id:[int(pred_landmark[1][id]), \
                    int(pred_landmark[0][id])] for id in range(19)}
                dir_pth = tdir(config['training']['runs_dir'] + f'/pseudo-labels/epoch_{epoch+1}')
                with open('{0}/{1:03d}.json'.format(dir_pth, ID), 'w') as f:
                    json.dump(inference_marks, f)
            
            ID += 1

        mre = self.evaluater.cal_metrics()
        return mre

    def test_for_single_net(self, net=None, epoch=None, train=""):
        self.evaluater.reset()
        if net is not None:
            self.model = net
        assert (hasattr(self, 'model'))
        ID = 1

        mode = "Test1" if train == "" else "Train"
        dataset_1 = Cephalometric(self.config['dataset']['pth'], mode, do_repeat=False)
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                       shuffle=False, num_workers=self.config['training']['num_workers'])
        self.Radius = dataset_1.Radius

        for img, _, __, ___, landmark_list in tqdm(self.dataloader_1):
            img = img.cuda()

            heatmap, regression_y, regression_x = self.model(img)

            gray_to_PIL(heatmap[0][1].cpu().detach()) \
                .save(tfilename('visuals', str(ID) + '_heatmap.png'))
            # Vote for the final accurate point
            # print(" ======================================\nlandmarks")
            # print(landmark_list)
            pred_landmark = heatmap2landmark(heatmap, self.Radius)

            self.evaluater.record(pred_landmark, landmark_list)

            # Optional Save viusal results
            image_pred = visualize(img, pred_landmark, landmark_list)
            image_pred.save(tfilename('visuals', str(ID) + '_pred.png'))

            ID += 1

        self.evaluater.cal_metrics()


def heatmap2landmark(heatmap, Radius):
    heatmap = heatmap.cpu()
    n, c, h, w = heatmap.shape
    num_candi = int(3.14 * Radius * Radius)
    # spots_heat, spots = heatmap.view(n, c, -1).topk(dim=-1, k=num_candi)
    spots_heat, spots = heatmap.view(n, c, -1).topk(dim=-1, k=1)
    spots_y = spots // w
    spots_x = spots % w
    spots_y = spots_y.view(-1).numpy()
    spots_x = spots_x.view(-1).numpy()
    # print("-------------------\n llist", llist)
    return [spots_y, spots_x]


if __name__ == "__main__":
    from tutils import trans_init, trans_args, tfilename
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train a cgan Xray network")
    parser.add_argument("--tag", default='test', help="position of the output dir")
    parser.add_argument("--config_file", default="config.yaml", help="default configs")
    parser.add_argument("--epoch", default=5, help="position of the output dir")
    args = trans_args(parser)
    logger, config = trans_init(args)

    # Load model
    net = UNet_Pretrained(3, config['training']['num_landmarks'])
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    logger.info("Loading checkpoints from epoch {}".format(args.epoch))
    checkpoints = torch.load(tfilename(config['base']['runs_dir'], \
                        "model_epoch_{}.pth".format(args.epoch)))
    net.load_state_dict(checkpoints)
    net = torch.nn.DataParallel(net)

    tester = Tester(logger, config, net, args.tag, args=args)
    tester.test(net)
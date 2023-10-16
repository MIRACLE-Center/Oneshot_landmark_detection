import argparse
import csv
import datetime
import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import os
import yaml
import yamlloader
import json

from .SAM_network import UNet_Pretrained
from datasets import select_dataset_SAM, select_dataset_SSL_Infer
from utils.mylogger import get_mylogger, set_logger_dir
from evaluation.eval import Evaluater
from config import get_cfg_defaults, merge_cfg_datasets, merge_cfg_train
from utils import to_Image, pred_landmarks, visualize, make_dir, np_rgb_to_PIL, show_cam_on_image


def match_cos(feature, template):
    feature = feature.permute(1, 2, 0)
    fea_L2 = torch.norm(feature, dim=-1)
    template_L2 = torch.norm(template, dim=-1)
    inner_product = (feature * template).sum(-1)
    cos_similarity = inner_product / (fea_L2 * template_L2 + 1e-3)
    return torch.clamp(cos_similarity, 0, 1)


class Tester(object):
    def __init__(self, logger, cfg, tag=None):
        self.cfg = cfg
        self.tag = tag
        
        # Creat evluater to record results
        self.evaluater = Evaluater(logger, cfg.dataset.num_landmarks, cfg.dataset.eval_radius)
        self.logger = logger

        self.id_landmarks = [i for i in range(cfg.dataset.num_landmarks)]

    def test(self, net, dump_label=False, mode='Test', id_shot=None):
        net.eval()
        runs_dir = self.cfg.runs_dir

        if id_shot is None:
            id_shot = self.cfg.test.id_oneshot[0]

        dataset_class = select_dataset_SSL_Infer(self.cfg.dataset.dataset)
        dataset = dataset_class(self.cfg.dataset.dataset_pth, mode, size=self.cfg.train.input_size)
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=self.cfg.train.num_workers)

        dataset_class = select_dataset_SSL_Infer(self.cfg.dataset.dataset)
        one_shot_loader = dataset_class(pathDataset=self.cfg.dataset.dataset_pth, \
            mode='Oneshot', size=self.cfg.train.input_size, id_oneshot=id_shot)
        
        self.logger.info(f'ID Oneshot : {id_shot}')
        self.evaluater.reset()
        image, patch_landmarks, template_patches, landmarks = one_shot_loader.__getitem__(0)
        # print(f"landmarks {landmarks}")
        feature_list = list()
        template_patches = template_patches.cuda()
        image = image.cuda()
        # import ipdb; ipdb.set_trace()
        # features = net_patch(template_patches)
        net_features_local, net_features_global = net(image.unsqueeze(0))
        # print(index, feature)

        
        tmp = list()
        for id_mark, landmark in enumerate(landmarks):
            tmpl_y, tmpl_x = landmark[0] // 2 , landmark[1] // 2
            # import ipdb; ipdb.set_trace()
            mark_feature = net_features_local[0, :, tmpl_y, tmpl_x]
            tmp.append(mark_feature.detach().squeeze().cpu().numpy())
        tmp = np.stack(tmp)
        feature_list_local = torch.tensor(tmp).cuda()

        tmp = list()
        for id_mark, landmark in enumerate(landmarks):
            tmpl_y, tmpl_x = landmark[0] // 2 ** 4, landmark[1] // 2 ** 4
            # import ipdb; ipdb.set_trace()
            mark_feature = net_features_global[0, :, tmpl_y, tmpl_x]
            tmp.append(mark_feature.detach().squeeze().cpu().numpy())
        tmp = np.stack(tmp)
        feature_list_global = torch.tensor(tmp).cuda()            

        for img, landmark_list, id_str, scale_rate in dataloader:
            img = img.cuda()
            features_local, features_global = net(img)
            id_str = id_str[0]
            scale_rate = [scale_rate[0][0].item(), scale_rate[1][0].item()]
            
            pred_landmarks_y, pred_landmarks_x = list(), list()
            for id_mark in range(feature_list_global.shape[0]):
                cos_lists = []
                final_cos = torch.ones_like(img[0,0]).cuda()
                
                cos_similarity = match_cos(features_local.squeeze(),\
                        feature_list_local[id_mark])
                cos_similarity = torch.nn.functional.upsample(\
                    cos_similarity.unsqueeze(0).unsqueeze(0), \
                    scale_factor=2**1, mode='bilinear').squeeze()
                final_cos = final_cos * cos_similarity

                cos_similarity = match_cos(features_global.squeeze(),\
                        feature_list_global[id_mark])
                cos_similarity = torch.nn.functional.upsample(\
                    cos_similarity.unsqueeze(0).unsqueeze(0), \
                    scale_factor=2**4, mode='bilinear').squeeze()
                # if id_depth > 1:
                #     cos_similarity = (cos_similarity + 0.2)
                final_cos = final_cos * cos_similarity

                final_cos = (final_cos - final_cos.min()) / \
                    (final_cos.max() - final_cos.min())

                chosen_landmark = final_cos.argmax().item()
                pred_landmarks_y.append(chosen_landmark // 512)
                pred_landmarks_x.append(chosen_landmark % 512)

                # if not os.path.isdir(f'test_visuals_SAM/{id_str}'):
                #     os.mkdir(f'test_visuals_SAM/{id_str}')
                # debug_visuals = [show_cam_on_image(img, cosmap) for cosmap in cos_lists]
                # debug = np.concatenate(debug_visuals, 1)
                # np_rgb_to_PIL(debug).save(os.path.join('test_visuals', id_str, f'{id_mark+1}_debug.jpg'))

            preds = [np.array(pred_landmarks_y), np.array(pred_landmarks_x)] 
            self.evaluater.record(preds, landmark_list, scale_rate)
            self.evaluater.save_img(img, preds, landmark_list, runs_dir, id_str)
            
            
            if dump_label:
                self.evaluater.save_preds(preds, runs_dir, id_str)
            
        self.evaluater.cal_metrics()


if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='gold', help="name of the run")
    parser.add_argument("--config_file", default="config.yaml", help="default configs")
    parser.add_argument("--dataset", default='head', choices=['head', 'hand', 'leg', 'chest'], help="dataset")
    parser.add_argument("--test", type=int, default=0, help="Test Mode")
    parser.add_argument("--epoch", type=int, default=299, help="Test Mode")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    merge_cfg_datasets(cfg, args.dataset)
    merge_cfg_train(cfg, 'SAM')

    # Create Logger
    logger = get_mylogger()
    

    # Create runs dir
    tag = args.tag
    runs_dir = "../../final_runs/" + args.tag
    runs_path = Path(runs_dir)
    cfg.runs_dir = runs_dir
    if not runs_path.exists():
        runs_path.mkdir()
    set_logger_dir(logger, runs_dir)
    logger.info(cfg)

    # Tester
    tester = Tester(logger, cfg, tag=args.tag)

    net = UNet_Pretrained(3, non_local=True)
    # net = torch.nn.DataParallel(net)
    net = net.cuda()
    net.eval()

    epoch = args.epoch
    ckpt = runs_dir + f"/model.pth"
    print(f'Load CKPT {ckpt}')
    ckpt = torch.load(ckpt)
    net.load_state_dict(ckpt)

    
    tester.test(net)
    
    
    # tester.test(net, net_patch, epoch=epoch, dump_label=False)
    # tester_train.test(net, net_patch, epoch=epoch, dump_label=True)

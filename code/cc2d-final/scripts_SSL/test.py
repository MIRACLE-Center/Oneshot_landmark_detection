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

from models import UNet_Pretrained
from datasets import select_dataset_SSL_Infer
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

        net_features = net(image.unsqueeze(0))
        # print(index, feature)

        feature_list = dict()
        for id_depth in range(6):
            tmp = list()
            for id_mark, landmark in enumerate(landmarks):
                tmpl_y, tmpl_x = landmark[0] // (2 ** (5-id_depth)), landmark[1] // (2 ** (5-id_depth))
                w, h = net_features[id_depth][0, 0].shape
                tmpl_y, tmpl_x = min(tmpl_y, w-1), min(tmpl_x, h-1)
                assert tmpl_y < w, f'template shape {w} chosen y {tmpl_y}'
                assert tmpl_x < h, f'template shape {h} chosen x {tmpl_x}'
                mark_feature = net_features[id_depth][0, :, tmpl_y, tmpl_x]
                tmp.append(mark_feature.detach().squeeze().cpu().numpy())
            tmp = np.stack(tmp)
            one_shot_feature = torch.tensor(tmp).cuda()
            feature_list[id_depth] = one_shot_feature

        for img, landmark_list, id_str, scale_rate in tqdm(dataloader):
            img = img.cuda()
            b, c, w, h = img.shape
            features = net(img)
            id_str = id_str[0]
            scale_rate = [scale_rate[0][0].item(), scale_rate[1][0].item()]
            
            pred_landmarks_y, pred_landmarks_x = list(), list()
            for id_mark in range(one_shot_feature.shape[0]):
                cos_lists = []
                final_cos = torch.ones_like(img[0,0]).cuda()
                for id_depth in range(5):
                    cos_similarity = match_cos(features[id_depth].squeeze(),\
                            feature_list[id_depth][id_mark])
                    cos_similarity = torch.nn.functional.upsample(\
                        cos_similarity.unsqueeze(0).unsqueeze(0), \
                        scale_factor=2**(5-id_depth), mode='bilinear').squeeze()
                    # if id_depth > 1:
                    #     cos_similarity = (cos_similarity + 0.2)
                    final_cos = final_cos * cos_similarity
                    cos_lists.append(cos_similarity)
                    # import ipdb; ipdb.set_trace()
                final_cos = (final_cos - final_cos.min()) / \
                    (final_cos.max() - final_cos.min())
                cos_lists.append(final_cos)
                chosen_landmark = final_cos.argmax().item()
                pred_landmarks_y.append(chosen_landmark // h)
                pred_landmarks_x.append(chosen_landmark % h)


                # pred_landmarks = torch.nonzero(final_cos == 1.0, as_tuple=False)[-1]
                # pred_landmarks_y.append(pred_landmarks[0].item())
                # pred_landmarks_x.append(pred_landmarks[1].item())

                # if not os.path.isdir(f'test_visuals/{id_str}'):
                #     os.mkdir(f'test_visuals/{id_str}')
                # debug_visuals = [show_cam_on_image(img, cosmap) for cosmap in cos_lists]
                # debug = np.concatenate(debug_visuals, 1)
                # np_rgb_to_PIL(debug).save(os.path.join('test_visuals', id_str, f'{id_mark+1}_debug.jpg'))

            preds = [np.array(pred_landmarks_y), np.array(pred_landmarks_x)] 
            self.evaluater.record(preds, landmark_list, scale_rate)
            self.evaluater.save_img(img, preds, landmark_list, runs_dir, id_str)
            
            
            if dump_label:
                self.evaluater.save_preds(preds, runs_dir, id_str)
            
        self.evaluater.cal_metrics()
        return self.evaluater.mre


if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='gold', help="name of the run")
    parser.add_argument("--dataset", default='head', help="dataset")
    parser.add_argument("--test", type=int, default=0, help="Test Mode")
    parser.add_argument("--epoch", type=int, default=0, help="Test Mode")
    parser.add_argument("--finetune", type=int, default=1, help="Test Mode")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    merge_cfg_datasets(cfg, args.dataset)
    merge_cfg_train(cfg, 'SSL')

    # Create Logger
    logger = get_mylogger()
    

    # Create runs dir
    tag = args.tag
    if args.finetune:
        runs_dir = "../../final_runs/" + tag + '/ssl_finetune'
    else:
        runs_dir = "../../final_runs/" + tag + '/ssl_scratch'
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
    ckpt = runs_dir + f"/model_epoch_current.pth"
    print(f'Load CKPT {ckpt}')
    ckpt = torch.load(ckpt)
    net.load_state_dict(ckpt)

    # res = []
    # for i in range(1, 551):
    #     res.append(tester.test(net, dump_label=True, id_shot=i))
    # import ipdb; ipdb.set_trace()
    
    tester.test(net, dump_label=True)
    tester.test(net, mode='Infer_Train', dump_label=True)
    
    
    # tester.test(net, net_patch, epoch=epoch, dump_label=False)
    # tester_train.test(net, net_patch, epoch=epoch, dump_label=True)

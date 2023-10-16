import torch
import numpy as np
import cv2
import os
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
from imgaug.augmentables import KeypointsOnImage


from models import UNet_Pretrained
from datasets import select_dataset_SSL_Infer
from config import get_cfg_defaults, merge_cfg_datasets, merge_cfg_train
from utils import *
from .test import match_cos


def select_descrete_points_by_distance(points, distance=100):
    """ Remove points within certain range """
    select_points = []
    def _select(point, selected):
        for s in selected:
            d = (point[0]-s[0])**2 + (point[1]-s[1])**2
            if d < distance:
                return False
        return True
    for point in points:
        if _select(point, select_points):
            select_points.append(point)
    return select_points

def gen_sift_points(cfg):

    dataset_class = select_dataset_SSL_Infer(cfg.dataset.dataset)
    dataset = dataset_class(cfg.dataset.dataset_pth, 'Infer_Train', size=cfg.train.input_size)

    path_sift = os.path.join(cfg.runs_dir, 'sift_points')
    cfg.path_sift = path_sift
    make_dir(path_sift)

    distance = 100
    contrastThreshold = 0.04
    edgeThreshold = 10
    if cfg.dataset.dataset == 'leg': 
        distance = 150
        contrastThreshold = 0.01
        edgeThreshold = 8
    if cfg.dataset.dataset == 'chest': 
        distance = 100
        contrastThreshold = 0.02
    if cfg.dataset.dataset == 'hand': 
        distance = 100
        contrastThreshold = 0.01

    for key, value in tqdm(dataset.images.items(), desc='Get SIFT points'):
        file_name = os.path.join(path_sift, f'{key}.npy')
        if os.path.isfile(file_name): continue

        vis_name = os.path.join(path_sift, f'{key}.png')

        array = np.array(value.convert('L'))
        array = cv2.resize(array, cfg.train.input_size[::-1])
        sift = cv2.SIFT_create(contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold)
        kp = sift.detect(array, None)
        landmark_list = []
        response_list = []
        for kpi in kp:
            response_list.append(kpi.response)
            landmark_list.append(list(kpi.pt))
        response_list = np.array(response_list)
        rank = np.argsort(response_list)[::-1]
        response_list = response_list[rank]
        landmark_list = np.array(landmark_list)
        landmark_list = landmark_list[rank]
        lm = landmark_list
        # Remove the points near borders
        kp = np.array(kp)[rank]
        def check(lm):
            if (lm[0] > 38.4 and lm[0] < cfg.train.input_size[1] - 38.4 \
                and lm[1] > 38.4 and cfg.train.input_size[0] - 38.4):
                return True
            else: return False

        ind = [check(lmi) for lmi in lm]
        kp = kp[ind]
        landmark_list = landmark_list[ind]
        response_list = response_list[ind]
        
        landmark_list = select_descrete_points_by_distance(landmark_list, distance)
        visual_gray_points(array, landmark_list).save(vis_name)

        np.save(file_name, np.array(landmark_list))

def gen_maxsim(cfg, net):
    path_maxsim = os.path.join(cfg.runs_dir, 'max_sim')
    cfg.path_sift = path_maxsim
    make_dir(path_maxsim)

    dataset_class = select_dataset_SSL_Infer(cfg.dataset.dataset)
    dataset = dataset_class(cfg.dataset.dataset_pth, 'Infer_Train', size=cfg.train.input_size)
    dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=cfg.train.num_workers)

    num_training = dataset.__len__()

    

    for i in range(num_training):
        max_all_list = []
        id_shot_str = dataset.list[i]['ID']
        file_name = os.path.join(path_maxsim, f'max_sim_{id_shot_str}.npy')
        if os.path.isfile(file_name): continue

        one_shot_set = dataset_class(pathDataset=cfg.dataset.dataset_pth, \
                mode='Infer_Train', size=cfg.train.input_size)
        img, landmark_list, id_str, scale_rate = one_shot_set.__getitem__(i)

        sift_landmarks = np.load(os.path.join(cfg.runs_dir, 'sift_points', f'{id_str}.npy'))
        sift_landmarks = np.around(sift_landmarks).astype(np.int32)
        
        feature_list = dict()
        img = img.cuda()
        net_features = net(img.unsqueeze(0))
        for id_depth in range(5):
            tmp = list()
            for id_mark, landmark in enumerate(sift_landmarks):
                tmpl_y, tmpl_x = landmark[1] // (2 ** (5-id_depth)), landmark[0] // (2 ** (5-id_depth))
                w, h = net_features[id_depth][0, 0].shape
                tmpl_y, tmpl_x = min(tmpl_y, w-1), min(tmpl_x, h-1)
                assert tmpl_y < w, f'template shape {w} chosen y {tmpl_y}'
                assert tmpl_x < h, f'template shape {h} chosen x {tmpl_x}'
                mark_feature = net_features[id_depth][0, :, tmpl_y, tmpl_x]
                tmp.append(mark_feature.detach().squeeze().cpu().numpy())
            tmp = np.stack(tmp)
            one_shot_feature = torch.tensor(tmp).cuda()
            feature_list[id_depth] = one_shot_feature

        
        for img, landmark_list, id_str, scale_rate in tqdm(dataloader, desc=f'Max Sim {id_shot_str}'):
            img = img.cuda()
            b, c, w, h = img.shape
            features = net(img)
            
            max_sample_list = []
            for id_mark in range(one_shot_feature.shape[0]):
                cos_lists = []
                max_landmark_list = []
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
                    max_landmark_list.append(cos_similarity.max().cpu().item())
                    # import ipdb; ipdb.set_trace()
                _max = final_cos.max().cpu().item()
                max_landmark_list.append(_max)
                max_sample_list.append(np.array(max_landmark_list))
            max_all_list.append(np.array(max_sample_list))
        np.save(file_name, np.array(max_all_list))

def select_ids(cfg):
    path_maxsim = os.path.join(cfg.runs_dir, 'max_sim')
    cfg.path_sift = path_maxsim

    dataset_class = select_dataset_SSL_Infer(cfg.dataset.dataset)
    dataset = dataset_class(cfg.dataset.dataset_pth, 'Infer_Train', size=cfg.train.input_size)

    num_training = dataset.__len__()

    max_list = []
    id_list = []
    for i in range(num_training):
        id_shot_str = dataset.list[i]['ID']
        file_name = os.path.join(path_maxsim, f'max_sim_{id_shot_str}.npy')
        if not os.path.isfile(file_name): 
            raise FileNotFoundError
        max_sim = np.load(file_name).max(-1)
        max_list.append(max_sim.mean())
        id_list.append(id_shot_str)
    max_list = np.array(max_list)
    top_10 = np.argsort(max_list)[-10:][::-1]
    for item in top_10: print(item+1, '\t', id_list[item])
    return top_10
        

if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='gold', help="name of the run")
    parser.add_argument("--dataset", default='head', help="dataset")
    parser.add_argument("--test", type=int, default=0, help="Test Mode")
    parser.add_argument("--epoch", type=int, default=0, help="Test Mode")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    merge_cfg_datasets(cfg, args.dataset)
    merge_cfg_train(cfg, 'SSL')


    # Create Logger
    logger = get_mylogger()

    # Create runs dir
    tag = args.tag
    runs_dir = "../../final_runs/" + tag + '/SCP'
    runs_path = Path(runs_dir)
    cfg.runs_dir = runs_dir
    if not runs_path.exists():
        runs_path.mkdir()
    set_logger_dir(logger, runs_dir)
    logger.info(cfg)

    net = UNet_Pretrained(3, non_local=True)
    # net = torch.nn.DataParallel(net)
    net = net.cuda()
    net.eval()

    ckpt = os.path.join("../../final_runs/", tag, 'ssl_scratch',\
                         f"model_epoch_current.pth")
    print(f'Load CKPT {ckpt}')
    ckpt = torch.load(ckpt)
    net.load_state_dict(ckpt)

    gen_sift_points(cfg)
    gen_maxsim(cfg, net)
    oneshot_ids = select_ids(cfg)

    from .test import Tester
    tester = Tester(logger, cfg, tag=args.tag)
    tester.test(net, dump_label=False, id_shot=cfg.test.id_oneshot[0])
    for id_shot in oneshot_ids:
        tester.test(net, dump_label=False, id_shot=id_shot+1)


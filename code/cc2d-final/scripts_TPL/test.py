import argparse
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import BCELoss
import os
import shutil
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

import torch.multiprocessing as mp

from models import UNet_Voting
from datasets import select_dataset_voting
from utils.mylogger  import get_mylogger, set_logger_dir
from evaluation.eval import Evaluater
from config import get_cfg_defaults, merge_cfg_datasets, merge_cfg_train
from utils import make_dir, voting

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

class Tester(object):
    def __init__(self, logger, cfg, runs_dir, args=None):
        
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


    def vote(self, id):
        tmp = torch.from_numpy(np.load(os.path.join(self.temp_dir, f'{id}.npy')))
        heatmap, regression_y, regression_x = tmp[:1], tmp[1:2], tmp[2:3]

        pred_landmark, score_map = voting(heatmap, regression_y, \
            regression_x, self.Radius, multi_process=False, analysis=True)
        
        # np.save(os.path.join(self.temp_dir, f'{id}_score.npy'), score_map)
        voting_uncert = score_map.max(-1).max(-1).tolist()
        landmark_list= self.dataloader_1.dataset.get_landmark_gt(id)
        
        with open(os.path.join(self.temp_dir, f'{id}_pred.pkl'), 'wb') as f:
            pickle.dump([pred_landmark, landmark_list], f)
        
        with open(os.path.join(self.pred_dir, f'{id}.pkl'), 'wb') as f:
            pickle.dump([pred_landmark, landmark_list], f)
        
        with open(os.path.join(self.uncertainty_dir, f'{id}.pkl'), 'wb') as f:
            pickle.dump(voting_uncert, f)
        
        with open(os.path.join(self.dump_dir, f'{id}.pkl'), 'wb') as f:
            pickle.dump([pred_landmark, landmark_list], f)
        

    def test(self, net, epoch=None, mode="", dump_label=False):
        self.temp_dir = os.path.join(self.runs_dir, 'temp')
        make_dir(self.temp_dir)

        self.evaluater.reset()

        dataset_class = select_dataset_voting(self.cfg.dataset.dataset)
        dataset = dataset_class(self.cfg.dataset.dataset_pth, mode='Test', \
                            size=self.cfg.train.input_size, pseudo=False)
        self.dataloader_1 = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=self.cfg.train.num_workers)
        self.Radius = dataset.Radius

        img_dict = dict()
        for img, _, __, ___, landmark_list, id_str, scale_rate in tqdm(self.dataloader_1):
            img = img.cuda()
            id_str = id_str[0]

            with torch.no_grad():
                heatmap, regression_y, regression_x = net(img)

            gray_to_PIL(heatmap[0][1].cpu().detach())\
                .save(os.path.join(self.visual_dir, id_str+'_heatmap.png'))
            
            np.save(os.path.join(self.temp_dir, f'{id_str}.npy'), \
                torch.cat([heatmap, regression_y, regression_x], 0).cpu().numpy())
            
            img_dict[id_str] = img.cpu()
        
        self.logger.info('Inference heatmap and offset map Done')
            # Vote for the final accurate point
        
        chosen_id = self.dataloader_1.dataset.list
        process_pool = list()
        for item in chosen_id:
            item = item['ID']
            process = mp.Process(target=self.vote, args=(item,))
            process_pool.append(process)
            process.start()
        
        for process in process_pool:
            process.join()
        self.logger.info('Voting Done')

        for item in chosen_id:
            item = item['ID']
            with open(os.path.join(self.temp_dir, f'{item}_pred.pkl'), 'rb') as f:
                pred_landmark, landmark_list = pickle.load(f)
            self.evaluater.record(pred_landmark, landmark_list, scale_rate)
            self.evaluater.save_img(img_dict[item], pred_landmark, landmark_list, self.runs_dir, item)

        self.evaluater.cal_metrics()
        self.logger.debug('Calculating metrics Done')

        shutil.rmtree(self.temp_dir)
        self.logger.debug('Remove Trash Done')


if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train a cgan Xray network")
    parser.add_argument("--tag", default='gold', help="position of the output dir")
    parser.add_argument("--dataset", default='head', choices=['head', 'hand', 'leg', 'chest'], help="dataset")
    parser.add_argument("--epoch", default=89, help="position of the output dir")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    merge_cfg_datasets(cfg, args.dataset)
    merge_cfg_train(cfg, 'TPL')


    # Create Logger
    logger = get_mylogger()
    

    # Create runs dir
    tag =  args.tag
    # runs_dir = "../../runs/" + tag
    runs_dir = "../../final_runs/" + tag + '/tpl_final'
    runs_path = Path(runs_dir)

    if not runs_path.exists():
        runs_path.mkdir()
    set_logger_dir(logger, runs_dir)
    cfg.runs_dir = runs_dir
    logger.info(cfg)

    # Load model
    net = UNet_Voting(3, cfg.dataset.num_landmarks)
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    logger.info("Loading checkpoints from epoch")
    checkpoints = torch.load(os.path.join(cfg.runs_dir, \
                        "model_epoch.pth"))
    net.load_state_dict(checkpoints)
    net = torch.nn.DataParallel(net)

    tester = Tester(logger, cfg, runs_dir, args=args)
    tester.test(net, mode='Test')
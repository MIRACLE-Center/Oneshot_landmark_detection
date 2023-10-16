import argparse
import datetime
import os
from pathlib import Path
import time
from torch.nn.modules.loss import CrossEntropyLoss
import yaml
import yamlloader
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch.nn import functional as F
from torch.nn import BCELoss

from .SAM_network import UNet_Pretrained
from datasets import select_dataset_SAM
from utils.mylogger import get_mylogger, set_logger_dir
from config import get_cfg_defaults, merge_cfg_datasets, merge_cfg_train, merge_from_args
from .SAM_test import Tester
from PIL import Image
import numpy as np
import random

# torch.autograd.set_detect_anomaly(True)

def cos_visual(tensor):
    tensor = torch.clamp(tensor, 0, 10)
    tensor = tensor  * 25.5
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images

def gray_to_PIL(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor  * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images


def ce_loss(cos_map):
    b, h = cos_map.shape
    total_loss = list()
    for id in range(b):
        cos_map[id] = cos_map[id].exp()
        gt_value = cos_map[id, 0].clone()
        chosen_patch = cos_map[id]
        id_loss = - torch.log(gt_value / chosen_patch.sum())
        total_loss.append(id_loss)
    total_loss = torch.stack(total_loss)
    return total_loss.mean()


def match_inner_product(feature, template):
    feature = feature.permute(0, 2, 1)
    fea_L2 = torch.norm(feature, dim=-1)
    template = template.unsqueeze(1)
    template_L2 = torch.norm(template, dim=-1)
    inner_product = (feature * template).sum(-1)
    cos_similarity = inner_product / (fea_L2 * template_L2 + 1e-3)
    # cos_similarity = torch.clamp(cos_similarity, 0., 1 - 1e-6)
    # print(cos_similarity.max(), cos_similarity.min())
    return cos_similarity

def random_all(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='layer2_0.001', help="name of the run")
    parser.add_argument("--dataset", default='head', help="dataset")
    parser.add_argument("--epoch", type=int, default=239, help="Test Mode")
    parser.add_argument("--add_bias", type=int, default=1, help="Test Mode")
    parser.add_argument("--mask_fine_loss", type=int, default=0, help="Test Mode")
    parser.add_argument("--clip", type=float, default=0.7, help="Test Mode")
    parser.add_argument("--strength", type=float, default=0.1, help="Test Mode")
    parser.add_argument("--temprature", type=float, default=10, help="Test Mode")
    parser.add_argument("--min_prob", type=float, default=0.2, help="Test Mode")
    parser.add_argument("--radius_ratio", type=float, default=0.05, help="Test Mode")
    parser.add_argument("--use_probmap", type=float, default=1, help="Test Mode")
    args = parser.parse_args()
    random_all(2022)

    cfg = get_cfg_defaults()
    merge_cfg_datasets(cfg, args.dataset)
    merge_cfg_train(cfg, 'SAM')

    merge_from_args(cfg, args, key_list=['tag', 'temprature', 'min_prob', 'radius_ratio'])

    # Create Logger
    logger = get_mylogger()
    

    # Create runs dir
    tag = str(datetime.datetime.now()).replace(' ', '-') if args.tag == '' else args.tag
    runs_dir = "../../final_runs/" + tag
    runs_path = Path(runs_dir)
    cfg.runs_dir = runs_dir
    if not runs_path.exists():
        runs_path.mkdir()
    set_logger_dir(logger, runs_dir)
    logger.info(cfg)

    # Tester
    tester = Tester(logger, cfg, tag=args.tag)

    dataset_class = select_dataset_SAM(cfg.dataset.dataset)
    dataset = dataset_class(cfg.dataset.dataset_pth, 'Train', use_probmap=args.use_probmap, \
        radius_ratio=args.radius_ratio, min_prob=args.min_prob, size=cfg.train.input_size, tag_ssl=args.tag)
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size,
                drop_last=True, shuffle=True, num_workers=cfg.train.num_workers)
    dataset.__getitem__(0)
    
    # net = UNet(3, config['num_landmarks'])
    # net = DeepLabv3_plus()
    net = UNet_Pretrained(3, non_local=True)

    # net = torch.nn.DataParallel(net)
    net = net.cuda()
    # net = net_patch
    # logger.info(net)

    optimizer = optim.Adam(params=net.parameters(), \
                           lr=cfg.train.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    scheduler = StepLR(optimizer, cfg.train.decay_step, gamma=cfg.train.decay_gamma)

    # loss
    loss_logic_fn = torch.nn.CrossEntropyLoss()
    rela_loss = torch.nn.MSELoss()

    net.eval()
    tester.test(net)

    for epoch in range(cfg.train.num_epochs):
        net.train()
        logic_loss_list = list()
        for index, (patch_1, patch_2, [chosen_y_1, chosen_x_1], [chosen_y_2, chosen_x_2], \
            [rand_global_y, rand_global_x], [rand_local_y, rand_local_x]) in enumerate(dataloader):

            patch_1 = patch_1.cuda()
            patch_2 = patch_2.cuda()
            
            raw_fea_local, raw_fea_global = net(patch_1)
            aug_fea_local, aug_fea_global = net(patch_2)
            

            # gray_to_PIL(raw_img[0,0,raw_y[0]:raw_y[0]+150,raw_x[0]:raw_x[0]+150].cpu()).save('img_before.jpg')
            # gray_to_PIL(crop_imgs[0,0,chosen_y[0]:chosen_y[0]+150,chosen_x[0]:chosen_x[0]+150].cpu()).save('img_after.jpg')
            # import ipdb; ipdb.set_trace()

            # loss base
            gt_y, gt_x = chosen_y_2 // (2 ** 4), chosen_x_2 // (2 ** 4)
            tmpl_y, tmpl_x = chosen_y_1 // (2 ** 4), chosen_x_1 // (2 ** 4)
            rand_global_y, rand_global_x = rand_global_y // (2 ** 4), rand_global_x // (2 ** 4)
            rand_global_y = torch.cat([gt_y.unsqueeze(1), rand_global_y.long()], -1)
            rand_global_x = torch.cat([gt_x.unsqueeze(1), rand_global_x.long()], -1)
            rand_global_ids = (rand_global_y * raw_fea_global.shape[-1] + rand_global_x).cuda()
            dim_feature = aug_fea_global.shape[1]

            tmpl_feature = torch.stack([raw_fea_global[id,:,tmpl_y[id], tmpl_x[id]]\
                for id in range(gt_y.shape[0])]).squeeze()
            aug_points_global = torch.stack([aug_fea_global[id].view(dim_feature, -1)\
                                            .index_select(-1, rand_global_ids[id]) \
                                            for id in range(gt_y.shape[0])]).squeeze()

            ret_inner_global = match_inner_product(aug_points_global, tmpl_feature) * args.temprature
            loss_global = ce_loss(ret_inner_global)


            # loss base
            gt_y, gt_x = chosen_y_2 // (2 ** 1), chosen_x_2 // (2 ** 1)
            tmpl_y, tmpl_x = chosen_y_1 // (2 ** 1), chosen_x_1 // (2 ** 1)
            rand_local_y, rand_local_x = rand_local_y // (2 ** 1), rand_local_x // (2 ** 1)
            rand_local_y = torch.cat([gt_y.unsqueeze(1), rand_local_y.long()], -1)
            rand_local_x = torch.cat([gt_x.unsqueeze(1), rand_local_x.long()], -1)
            rand_local_ids = (rand_local_y * raw_fea_local.shape[-1] + rand_local_x).cuda()
            dim_feature = aug_fea_local.shape[1]

            tmpl_feature = torch.stack([raw_fea_local[id,:,tmpl_y[id], tmpl_x[id]]\
                for id in range(gt_y.shape[0])]).squeeze()
            aug_points_local = torch.stack([aug_fea_local[id].view(dim_feature, -1)\
                                            .index_select(-1, rand_local_ids[id]) \
                                            for id in range(gt_y.shape[0])]).squeeze()

            ret_inner_local = match_inner_product(aug_points_local, tmpl_feature) * args.temprature
            loss_local = ce_loss(ret_inner_local)
            loss_compare = loss_local + loss_global
            
            optimizer.zero_grad()
            loss_compare.backward()
            optimizer.step()

            logic_loss_list.append(np.array([loss_local.cpu().item(), loss_global.cpu().item()]))
        
        # cos_visual(ret_inner_5[0].cpu()).save('layer_5.jpg')
        # cos_visual(ret_inner_4[0].cpu()).save('layer_4.jpg')
        # cos_visual(ret_inner_3[0].cpu()).save('layer_3.jpg')
        # cos_visual(ret_inner_2[0].cpu()).save('layer_2.jpg')
        # cos_visual(ret_inner_1[0].cpu()).save('layer_1.jpg')
        losses = np.stack(logic_loss_list).transpose()
        # import ipdb; ipdb.set_trace()
        logger.info("Epoch {} Training logic loss local {:.3f} global {:.3f}". \
                    format(epoch, losses[0].mean(), losses[1].mean()))

        scheduler.step()

        # # for img, guassian_mask, landmark_list in tqdm(tester.dataloader_1):
        # #     img, guassian_mask = img.cuda(), guassian_mask.cuda()
        # #     with torch.no_grad():
        # #         heatmap = net(img)
        # #     loss = loss_logic_fn(heatmap, guassian_mask)

        # #     logic_loss_list.append(loss.cpu().item())
        # # logger.info("Epoch {} Testing logic loss {}". \
        # #             format(epoch, sum(logic_loss_list) / tester.dataset.__len__()))
        # # save model
        if (epoch + 1) % cfg.train.save_seq == 0:
        # if True:
            net.eval()
            torch.save(net.state_dict(), runs_dir + "/model.pth")

            tester.test(net, dump_label=False)
            logger.info(runs_dir + "/model_epoch_{}.pth".format(epoch))
            
            cfg.train.last_epoch = epoch
            # tester.test(net, epoch=epoch)
       # dump yaml
        with open(runs_dir + "/config.txt", "w") as f:
            f.write(cfg.dump())

    # # Test
    # tester.test(net)

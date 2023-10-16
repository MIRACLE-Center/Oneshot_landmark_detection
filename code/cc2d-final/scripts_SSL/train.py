import argparse
import datetime
import os
from pathlib import Path
import yaml
import yamlloader
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim

from models.network import UNet_Pretrained
from datasets import select_dataset_SSL_Train
from config import get_cfg_defaults, merge_cfg_datasets, merge_cfg_train, merge_from_args
from utils.mylogger import get_mylogger, set_logger_dir
from .test import Tester
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


def ce_loss(cos_map, gt_y, gt_x, nearby=None, add_bias=True, args=None):
    b, h, w = cos_map.shape
    total_loss = list()
    for id in range(b):
        # cos_map[id] = cos_map[id].exp()
        # gt_value = cos_map[id, gt_y[id], gt_x[id]].clone()
        if nearby is not None:
            min_y, max_y = max(gt_y[id] - nearby, 0), min(gt_y[id] + nearby, h)
            min_x, max_x = max(gt_x[id] - nearby, 0), min(gt_x[id] + nearby, w)
            chosen_patch = cos_map[id, min_y:max_y, min_x:max_x]
            gap_y, gap_x = gt_y[id] - min_y, gt_x[id] - min_x
            y, x = chosen_patch.shape
            mesh = torch.meshgrid(torch.linspace(0, y-1, y), torch.linspace(0, x-1, x))
            distance_y, distance_x = mesh[0] - gap_y, mesh[1] - gap_x
            distance = torch.sqrt(distance_y**2 + distance_x**2).cuda()
            distance = torch.clamp(distance*args.strength * args.temprature, 0, args.clip * args.temprature)
            if add_bias:
                chosen_patch += distance
            chosen_patch = chosen_patch.exp()
            gt_value = chosen_patch[gap_y, gap_x].clone()
        else:
            y, x = cos_map[id].shape
            mesh = torch.meshgrid(torch.linspace(0, y-1, y), torch.linspace(0, x-1, x))
            distance_y, distance_x = mesh[0] - gt_y[id], mesh[1] - gt_x[id]
            distance = torch.sqrt(distance_y**2 + distance_x**2).cuda()
            distance = torch.clamp(distance*args.strength * args.temprature, 0, args.clip * args.temprature)
            if add_bias:
                cos_map[id] += distance
            cos_map[id] = cos_map[id].exp()
            gt_value = cos_map[id, gt_y[id], gt_x[id]].clone()
            chosen_patch = cos_map[id]
        id_loss = - torch.log(gt_value / chosen_patch.sum())
        total_loss.append(id_loss)
    # print(torch.stack(total_loss).mean())
    return torch.stack(total_loss).mean()


def match_inner_product(feature, template):
    feature = feature.permute(0, 2, 3, 1)
    template = template.unsqueeze(1).unsqueeze(1)
    fea_L2 = torch.norm(feature, dim=-1)
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
    parser.add_argument("--config_file", default="config.yaml", help="default configs")
    parser.add_argument("--epoch", type=int, default=239, help="Test Mode")
    parser.add_argument("--add_bias", type=int, default=1, help="Test Mode")
    parser.add_argument("--mask_fine_loss", type=int, default=0, help="Test Mode")
    parser.add_argument("--clip", type=float, default=0.8, help="Test Mode")
    parser.add_argument("--strength", type=float, default=0.07, help="Test Mode")
    parser.add_argument("--temprature", type=float, default=10, help="Test Mode")
    parser.add_argument("--min_prob", type=float, default=0.2, help="Test Mode")
    parser.add_argument("--radius_ratio", type=float, default=0.05, help="Test Mode")
    parser.add_argument("--nearby", type=int, default=11, help="Test Mode")
    parser.add_argument("--length_embedding", type=int, default=64, help="Test Mode")
    parser.add_argument("--use_probmap", type=float, default=0, help="Test Mode")
    args = parser.parse_args()
    random_all(2022)

    cfg = get_cfg_defaults()
    merge_cfg_datasets(cfg, args.dataset)
    merge_cfg_train(cfg, 'SSL')

    merge_from_args(cfg, args, key_list=['nearby', 'add_bias', 'clip', 'strength', 'tag',\
                                'temprature', 'min_prob', 'radius_ratio', 'length_embedding'])
    
    # Create Logger
    logger = get_mylogger()
    

    # Create runs dir
    tag = str(datetime.datetime.now()).replace(' ', '-') if args.tag == '' else args.tag
    tag_ssl = 'ssl_scratch' if not args.use_probmap else 'ssl_finetune'
    runs_dir = os.path.join("../../final_runs/", tag)
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        runs_path.mkdir()
    runs_dir = os.path.join("../../final_runs/", tag, tag_ssl)
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        runs_path.mkdir()
    cfg.runs_dir = runs_dir
    set_logger_dir(logger, runs_dir)
    logger.info(cfg)

    # Tester
    tester = Tester(logger, cfg, tag=args.tag)

    dataset_class = select_dataset_SSL_Train(cfg.dataset.dataset)
    dataset = dataset_class(cfg.dataset.dataset_pth, 'Train', use_probmap=args.use_probmap, \
        radius_ratio=args.radius_ratio, min_prob=args.min_prob, size=cfg.train.input_size, tag_ssl=args.tag)
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size,
                drop_last=True, shuffle=True, num_workers=cfg.train.num_workers)
    dataset.__getitem__(0)
    # net = UNet(3, config['num_landmarks'])
    # net = DeepLabv3_plus()
    net = UNet_Pretrained(3, non_local=True, length_embedding=args.length_embedding)
    net_patch = UNet_Pretrained(3, non_local=False, length_embedding=args.length_embedding)

    # net = torch.nn.DataParallel(net)
    net = net.cuda()
    net_patch = net_patch.cuda()
    # net = net_patch
    # logger.info(net)

    # epoch = args.epoch
    # ckpt = runs_dir + f"/model_epoch_{epoch}.pth"
    # print(f'Load CKPT {ckpt}')
    # ckpt = torch.load(ckpt)
    # net.load_state_dict(ckpt)
    # ckpt = runs_dir + f"/model_patch_epoch_{epoch}.pth"
    # print(f'Load CKPT {ckpt}')
    # ckpt = torch.load(ckpt)1780
    # net_patch.load_state_dict(ckpt)

    optimizer = optim.Adam(params=net.parameters(), \
                           lr=cfg.train.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    scheduler = StepLR(optimizer, cfg.train.decay_step, gamma=cfg.train.decay_gamma)

    optimizer_patch = optim.Adam(params=net_patch.parameters(), \
                           lr=cfg.train.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    scheduler_patch = StepLR(optimizer_patch, cfg.train.decay_step, gamma=cfg.train.decay_gamma)

    # loss
    loss_logic_fn = torch.nn.CrossEntropyLoss()
    rela_loss = torch.nn.MSELoss()

    # Only for check dataset
    # for i in tqdm(range(dataset.__len__()), desc='Check datasets'):
    #     _ = dataset.__getitem__(i)

    net.eval()
    tester.test(net, dump_label=False)

    for epoch in range(cfg.train.num_epochs):
        net.train()
        net_patch.train()
        logic_loss_list = list()
        for index, (raw_img, crop_imgs, chosen_y, chosen_x, raw_y, raw_x) in enumerate(dataloader):
            
            raw_img = raw_img.cuda()
            crop_imgs = crop_imgs.cuda()
            
            raw_fea_list = net(raw_img)
            # crop_fea_list = net_patch(crop_imgs)
            crop_fea_list = net(crop_imgs)

            # gray_to_PIL(raw_img[0,0,raw_y[0]:raw_y[0]+150,raw_x[0]:raw_x[0]+150].cpu()).save('img_before.jpg')
            # gray_to_PIL(crop_imgs[0,0,chosen_y[0]:chosen_y[0]+150,chosen_x[0]:chosen_x[0]+150].cpu()).save('img_after.jpg')
            # import ipdb; ipdb.set_trace()

            # loss base
            gt_y, gt_x = raw_y // (2 ** 5), raw_x // (2 ** 5)
            tmpl_y, tmpl_x = chosen_y // (2 ** 5), chosen_x // (2 ** 5)
            tmpl_feature = torch.stack([crop_fea_list[0][[id],:,tmpl_y[id], tmpl_x[id]]\
                for id in range(gt_y.shape[0])]).squeeze()
            ret_inner_5 = match_inner_product(raw_fea_list[0], tmpl_feature) * args.temprature
            loss_5 = ce_loss(ret_inner_5, gt_y, gt_x, add_bias=args.add_bias, args=args)

            # loss base
            gt_y, gt_x = raw_y // (2 ** 4), raw_x // (2 ** 4)
            tmpl_y, tmpl_x = chosen_y // (2 ** 4), chosen_x // (2 ** 4)
            tmpl_feature = torch.stack([crop_fea_list[1][[id],:,tmpl_y[id], tmpl_x[id]]\
                for id in range(gt_y.shape[0])]).squeeze()
            ret_inner_4 = match_inner_product(raw_fea_list[1], tmpl_feature) * args.temprature
            loss_4 = ce_loss(ret_inner_4, gt_y, gt_x, nearby=cfg.args.nearby, add_bias=args.add_bias, args=args)

            # loss base
            gt_y, gt_x = raw_y // (2 ** 3), raw_x // (2 ** 3)
            tmpl_y, tmpl_x = chosen_y // (2 ** 3), chosen_x // (2 ** 3)
            tmpl_feature = torch.stack([crop_fea_list[2][[id],:,tmpl_y[id], tmpl_x[id]]\
                for id in range(gt_y.shape[0])]).squeeze()
            ret_inner_3 = match_inner_product(raw_fea_list[2], tmpl_feature) * args.temprature
            loss_3 = ce_loss(ret_inner_3, gt_y, gt_x, nearby=cfg.args.nearby, add_bias=args.add_bias, args=args)

            # loss base
            gt_y, gt_x = raw_y // (2 ** 2), raw_x // (2 ** 2)
            tmpl_y, tmpl_x = chosen_y // (2 ** 2), chosen_x // (2 ** 2)
            tmpl_feature = torch.stack([crop_fea_list[3][[id],:,tmpl_y[id], tmpl_x[id]]\
                for id in range(gt_y.shape[0])]).squeeze()
            ret_inner_2 = match_inner_product(raw_fea_list[3], tmpl_feature) * args.temprature
            loss_2 = ce_loss(ret_inner_2, gt_y, gt_x, nearby=cfg.args.nearby, add_bias=args.add_bias, args=args)

            # loss base
            gt_y, gt_x = raw_y // (2 ** 1), raw_x // (2 ** 1)
            tmpl_y, tmpl_x = chosen_y // (2 ** 1), chosen_x // (2 ** 1)
            tmpl_feature = torch.stack([crop_fea_list[4][[id],:,tmpl_y[id], tmpl_x[id]]\
                for id in range(gt_y.shape[0])]).squeeze()
            ret_inner_1 = match_inner_product(raw_fea_list[4], tmpl_feature) * args.temprature
            loss_1 = ce_loss(ret_inner_1, gt_y, gt_x, nearby=cfg.args.nearby, add_bias=args.add_bias, args=args)

            loss_compare = loss_5 + loss_4 + loss_3 + loss_2 + loss_1
            
            loss = loss_compare

            optimizer.zero_grad()
            # optimizer_patch.zero_grad()
            loss.backward()
            optimizer.step()
            # optimizer_patch.step()

            logic_loss_list.append(np.array([loss_5.cpu().item(), loss_4.cpu().item(), \
                loss_3.cpu().item(), loss_2.cpu().item(), loss_1.cpu().item(), 0]))
        
        # cos_visual(ret_inner_5[0].cpu()).save('layer_5.jpg')
        # cos_visual(ret_inner_4[0].cpu()).save('layer_4.jpg')
        # cos_visual(ret_inner_3[0].cpu()).save('layer_3.jpg')
        # cos_visual(ret_inner_2[0].cpu()).save('layer_2.jpg')
        # cos_visual(ret_inner_1[0].cpu()).save('layer_1.jpg')
        losses = np.stack(logic_loss_list).transpose()
        # import ipdb; ipdb.set_trace()
        logger.info("Epoch {} Training logic loss 5 {:.3f} 4 {:.3f} 3 {:.3f} 2 {:.3f} 1 {:.3f} rela {:.3f}". \
                    format(epoch, losses[0].mean(), losses[1].mean(), losses[2].mean(),\
                        losses[3].mean(), losses[4].mean(), losses[5].mean()))

        scheduler.step()
        scheduler_patch.step()

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
            net_patch.eval()
            tester.test(net, dump_label=False)
            logger.info(runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), runs_dir + "/model_epoch_current.pth".format(epoch))
            torch.save(net_patch.state_dict(), runs_dir + "/model_patch_epoch_current.pth".format(epoch))

            cfg.train.last_epoch = epoch
            # tester.test(net, epoch=epoch)
       # dump yaml
        with open(runs_dir + "/config.txt", "w") as f:
            f.write(cfg.dump())
    
    net.eval()
    net_patch.eval()
    tester.test(net, dump_label=True, mode='Infer_Train')

    # # Test
    # tester.test(net)

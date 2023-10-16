import argparse
import datetime
import os
from pathlib import Path
import time
import yaml
import yamlloader
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch.nn import functional as F
from torch.nn import BCELoss

from models import UNet_Voting
from datasets import select_dataset_voting
from utils.mylogger import get_mylogger, set_logger_dir
from .test import Tester
from config import get_cfg_defaults, merge_cfg_datasets, merge_cfg_train
from PIL import Image
import numpy as np

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
    parser.add_argument("--stage_probmap", type=float, default=1, help="Test Mode")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    merge_cfg_datasets(cfg, args.dataset)
    merge_cfg_train(cfg, 'TPL')

    # Create Logger
    logger = get_mylogger()
    logger.info(cfg)

    # Create runs dir
    tag = str(datetime.datetime.now()).replace(' ', '-') if args.tag == '' else args.tag
    # tag_tpl = 'tpl_scratch' if not args.stage_probmap else 'tpl_final'
    tag_ssl = 'ssl_scratch' if not args.stage_probmap else 'ssl_finetune'

    ssl_dir = os.path.join("../../final_runs/", tag, tag_ssl)
    runs_path_ssl = Path(ssl_dir)
    if not runs_path_ssl.exists():
        assert f"No SSL runs dir found, tag {runs_path_ssl}"
    runs_dir = os.path.join("../../final_runs/", tag, tag_tpl)
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        runs_path.mkdir()
    if not runs_path.exists():
        runs_path.mkdir()
    set_logger_dir(logger, runs_dir)

    cfg.tag = tag
    cfg.runs_dir = runs_dir


    dataset_class = select_dataset_voting(args.dataset)
    oneshot_dataset = dataset_class(cfg.dataset.dataset_pth, 'Oneshot', size=cfg.train.input_size, id_shot=cfg.test.id_oneshot[0])
    oneshot_dataloader = DataLoader(oneshot_dataset, batch_size=1,
                            shuffle=True, num_workers=1)

    # net = UNet(3, config['num_landmarks'])
    net = UNet_Voting(3, cfg.dataset.num_landmarks, non_local=False)
    net = torch.nn.DataParallel(net)
    net = net.cuda()
    # logger.info(net)

    ema_net = UNet_Voting(3, cfg.dataset.num_landmarks, non_local=False)
    ema_net = torch.nn.DataParallel(ema_net)
    ema_net = ema_net.cuda()
    for param in ema_net.parameters():
        param.detach_()


    start_epoch = 0
    if args.epoch > 0:
        start_epoch = args.epoch + 1
        logger.info("Loading checkpoints from epoch {}".format(args.epoch))
        checkpoints = torch.load(os.path.join(cfg.runs_dir, \
                            "model_epoch_{}.pth".format(args.epoch)))
        net.load_state_dict(checkpoints)

    optimizer = optim.Adam(params=net.parameters(), \
                           lr=cfg.train.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    scheduler = StepLR(optimizer, cfg.train.decay_step, gamma=cfg.train.decay_gamma)

    # loss
    # weight = torch.tensor([1,1,1,1,1,1,1,1]).float().cuda()
    # loss_logic_fn = BCELoss(reduction='none')
    loss_logic_fn = focal_loss
    loss_regression_fn = L1Loss

    # Tester
    tester = Tester(logger, cfg, runs_dir=runs_dir)

    oneshot_generater = iter(oneshot_dataloader)
    shot_img, shot_mask, shot_offset_y, shot_offset_x, _, _, _, = \
        next(oneshot_generater)

    num_iter = 0
    for epoch in range(start_epoch, cfg.train.num_epochs):
        logic_loss_list = list()
        net.train()

        dataset_class = select_dataset_voting(args.dataset)
        dataset = dataset_class(cfg.dataset.dataset_pth, mode='Train', size=cfg.train.input_size, ssl_dir=ssl_dir)
        dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size - 1,
                    drop_last=True, shuffle=True, num_workers=cfg.train.num_workers)
        for img, mask, offset_y, offset_x, landmark_list, _, __ in tqdm(dataloader):

            img = torch.cat([img, shot_img], 0)
            mask = torch.cat([mask, shot_mask], 0)
            offset_y = torch.cat([offset_y, shot_offset_y], 0)
            offset_x = torch.cat([offset_x, shot_offset_x], 0)

            img, mask, offset_y, offset_x = img.cuda(), \
                mask.cuda(), offset_y.cuda(), offset_x.cuda()
            
            heatmap, regression_y, regression_x = net(img)

            logic_loss = loss_logic_fn(heatmap, mask)
            regression_loss_y = loss_regression_fn(regression_y, offset_y, mask)
            regression_loss_x = loss_regression_fn(regression_x, offset_x, mask)

            loss = regression_loss_x + regression_loss_y + logic_loss * cfg.train.loss_lambda
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

        # # save model
        if (epoch + 1) % cfg.train.save_seq == 0:

            logger.info(runs_dir + "/model_epoch.pth")
            torch.save(ema_net.state_dict(), runs_dir + "/model_epoch.pth")

            cfg.train.last_epoch = epoch
            ema_net.eval()
            tester.test(ema_net, epoch=epoch)

        # net.eval()
        # tester.test(net, epoch=epoch, train="Train", dump_label=True)

        # dump yaml
        with open(runs_dir + "/config.txt", "w") as f:
            f.write(cfg.dump())

    # Test
    ema_net.eval()
    # import ipdb; ipdb.set_trace()
    tester.test(ema_net, mode='Infer_train')

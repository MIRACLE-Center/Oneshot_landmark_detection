import argparse
import os
import yaml
import yamlloader
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
from models.network_std import UNet_Pretrained
from datasets.ceph_st import Cephalometric
from .test_st import Tester
from PIL import Image
import numpy as np

from tutils import trans_init, trans_args, dump_yaml, tfuncname, tfilename


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
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images


def focal_loss(pred, gt):
    return (-(1 - pred) * gt * torch.log(pred) - pred * (1 - gt) * torch.log(1 - pred)).mean()


def dump_best_config(logger, config, info):
    # dump yaml
    config = {**config, **info}
    with open(config['runs_dir'] + "/best_config.yaml", "w") as f:
        yaml.dump(config, f)
    logger.info("Dump best config")


if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='', help="name of the run")
    parser.add_argument("--config", default="configs/config.yaml", help="default configs")
    parser.add_argument("--epoch", type=int, default=0, help="default configs")
    parser.add_argument("--pseudo", type=str, default="debug")
    parser.add_argument("--oneshot", type=int, default=126)
    parser.add_argument('--finaltest', action='store_true')
    args = parser.parse_args()
    logger, config = trans_init(args)

    pseudo_config_path = config['pseudo_config_path'] = \
        tfilename(config['base']['base_dir'], config['base']['experiment'], args.pseudo + '/config.yaml')
    with open(pseudo_config_path, 'r') as f:
        pseudo_config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
        pseudo_config['oneshot_id'] = args.oneshot
    assert pseudo_config is not None

    oneshot_dataset = Cephalometric(config['dataset']['pth'], 'Oneshot')
    oneshot_dataloader = DataLoader(oneshot_dataset, batch_size=1,
                                    shuffle=True, num_workers=config['training']['num_workers'])

    # net = UNet(3, config['num_landmarks'])
    net = UNet_Pretrained(3, config['training']['num_landmarks'])
    net = torch.nn.DataParallel(net)
    net = net.cuda()
    # logger.info(net)

    start_epoch = 0
    if args.epoch > 0:
        start_epoch = args.epoch + 1
        logger.info("Loading checkpoints from epoch {}".format(args.epoch))
        checkpoints = torch.load(os.path.join(config['training']['runs_dir'], \
                                              "model_epoch_{}.pth".format(args.epoch)))
        net.load_state_dict(checkpoints)

    optimizer = optim.Adam(params=net.parameters(), \
                           lr=config['training']['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    scheduler = StepLR(optimizer, config['training']['decay_step'], gamma=config['training']['decay_gamma'])

    loss_logic_fn = focal_loss
    loss_regression_fn = L1Loss

    # Tester
    tester = Tester(logger, config, test_mode=2)

    # Record
    best_mre = 100.0
    best_epoch = -1

    for epoch in range(start_epoch, config['training']['num_epochs']):
        logic_loss_list = list()
        net.train()

        select_epoch = epoch if epoch > 100 else 0
        dataset = Cephalometric(config['dataset']['pth'], mode='Train', epoch=select_epoch, config=config, pseudo_config=pseudo_config)
        dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'] - 1,
                                drop_last=True, shuffle=True, num_workers=config['training']['num_workers'])
        for img, mask, offset_y, offset_x, landmark_list in tqdm(dataloader, ncols=100):
            oneshot_generater = iter(oneshot_dataloader)
            shot_img, shot_mask, shot_offset_y, shot_offset_x, _ = \
                next(oneshot_generater)

            img = torch.cat([img, shot_img], 0)
            mask = torch.cat([mask, shot_mask], 0)
            offset_y = torch.cat([offset_y, shot_offset_y], 0)
            offset_x = torch.cat([offset_x, shot_offset_x], 0)

            img, mask, offset_y, offset_x = img.cuda(), \
                                            mask.cuda(), offset_y.cuda(), offset_x.cuda()
            # import ipdb; ipdb.set_trace()

            heatmap, regression_y, regression_x = net(img)

            logic_loss = loss_logic_fn(heatmap, mask)
            regression_loss_y = loss_regression_fn(regression_y, offset_y, mask)
            regression_loss_x = loss_regression_fn(regression_x, offset_x, mask)

            loss = regression_loss_x + regression_loss_y + logic_loss * config['training']['lambda']
            # loss = (loss * weight).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logic_loss_list.append(loss.cpu().item())
        logger.info("Epoch {} Training logic loss {} ". \
                    format(epoch, sum(logic_loss_list) / dataset.__len__()))
        scheduler.step()

        # # save model
        if epoch == -1 or (epoch + 1) % config['training']['save_seq'] == 0:
            logger.info(config['training']['runs_dir'] + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), config['training']['runs_dir'] + "/model_epoch_{}.pth".format(epoch))

            config['training']['last_epoch'] = epoch
            net.eval()
            mre = tester.test(net, epoch=epoch)
            if mre < best_mre:
                best_mre = mre
                best_epoch = epoch
                save_dict = {"epoch": best_epoch, "mre": best_mre,
                             "model": config['training']['runs_dir'] + "/model_epoch_{}.pth".format(epoch),
                             "model_patch": config['training']['runs_dir'] + "/model_patch_epoch_{}.pth".format(epoch),
                             }
            logger.info(f"********  Best MRE:{best_mre} in Epoch {best_epoch} || Epoch{epoch}:{mre} ********")

        # dump yaml
        dump_yaml(logger, config)


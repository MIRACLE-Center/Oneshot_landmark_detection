"""
    Fixed some bugs:
        test with only net (no net_patch
        clip CosSimilarity to 0~1
        more narrow border when selected in 'Dataloader'
"""

import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
from models.network import UNet_Pretrained
from datasets.ceph import Cephalometric
from .test import Tester
from PIL import Image
import numpy as np

from tutils import trans_init, trans_args, dump_yaml, tfilename

# torch.autograd.set_detect_anomaly(True)

def cos_visual(tensor):
    tensor = torch.clamp(tensor, 0, 10)
    tensor = tensor * 25.5
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images


def gray_to_PIL(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images


def ce_loss(cos_map, gt_y, gt_x, nearby=None):
    b, h, w = cos_map.shape
    total_loss = list()
    for id in range(b):
        cos_map[id] = cos_map[id].exp()
        gt_value = cos_map[id, gt_y[id], gt_x[id]].clone()
        if nearby is not None:
            min_y, max_y = max(gt_y[id] - nearby, 0), min(gt_y[id] + nearby, h)
            min_x, max_x = max(gt_x[id] - nearby, 0), min(gt_x[id] + nearby, w)
            chosen_patch = cos_map[id, min_y:max_y, min_x:max_x]
        else:
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
    # print(cos_similarity.max(), cos_similarity.min())
    cos_similarity = torch.clamp(cos_similarity, 0., 1.)
    assert torch.max(cos_similarity) <= 1.0, f"Maximum Error, Got max={torch.max(cos_similarity)}"
    assert torch.min(cos_similarity) >= 0.0, f"Maximum Error, Got max={torch.min(cos_similarity)}"
    return cos_similarity 


cosfn = torch.nn.CosineSimilarity(dim=1, eps=1e-3)
def match_inner_product2(feature, template):
    cos_sim = cosfn(template.unsqueeze(-1).unsqueeze(-1), feature)
    return cos_sim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='', help="name of the run")
    parser.add_argument("--config", default="configs/config.yaml", help="default configs")
    parser.add_argument("--test", type=int, default=0, help="Test Mode")
    parser.add_argument("--resume", type=int , default=0)
    parser.add_argument("--pretrain", type=str, default="") #emb-16-289
    args = trans_args(parser)
    logger, config = trans_init(args)
    args = parser.parse_args()

    tag = config['base']['tag']
    runs_dir = config['base']['runs_dir']

    # Tester
    tester = Tester(logger, config, tag=args.tag)

    dataset = Cephalometric(config['dataset']['pth'], 'Train', patch_size=32*8, retfunc=1, use_prob=False)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'],
                            drop_last=True, shuffle=True, num_workers=config['training']['num_workers'])

    net = UNet_Pretrained(3, non_local=config['training']['non_local'], emb_len=16)
    logger.info(f"debug train2.py non local={config['training']['non_local']}")
    net_patch = UNet_Pretrained(3, emb_len=16)

    if args.pretrain is not None and args.pretrain != "":
        print("Pretrain emb-16 289")
        epoch = 289
        ckpt = "/data/quanquan/oneshot/runs2/" + "emb-16" + f"/model_epoch_{epoch}.pth"
        assert os.path.exists(ckpt)
        logger.info(f'Load CKPT {ckpt}')
        ckpt = torch.load(ckpt)
        net.load_state_dict(ckpt)
        ckpt2 = "/data/quanquan/oneshot/runs2/" + "emb-16" + f"/model_patch_epoch_{epoch}.pth"
        assert os.path.exists(ckpt2)
        ckpt2 = torch.load(ckpt2)
        net_patch.load_state_dict(ckpt2)

    if args.resume > 0:
        epoch = args.resume
        ckpt = runs_dir + f"/model_epoch_{epoch}.pth"
        assert os.path.exists(ckpt)
        logger.info(f'Load CKPT {ckpt}')
        ckpt = torch.load(ckpt)
        net.load_state_dict(ckpt)
        ckpt2 = runs_dir + f"/model_patch_epoch_{epoch}.pth"
        assert os.path.exists(ckpt2)
        ckpt2 = torch.load(ckpt2)
        net_patch.load_state_dict(ckpt2)

    net = net.cuda()
    net_patch = net_patch.cuda()

    if args.test:
        epoch = 109
        ckpt = runs_dir + f"/model_epoch_{epoch}.pth"
        print(f'Load CKPT {ckpt}')
        ckpt = torch.load(ckpt)
        net.load_state_dict(ckpt)
        tester.test(net, epoch=epoch)
        exit()

    optimizer = optim.Adam(params=net.parameters(), \
                           lr=config['training']['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    scheduler = StepLR(optimizer, config['training']['decay_step'], gamma=config['training']['decay_gamma'])

    optimizer_patch = optim.Adam(params=net_patch.parameters(), \
                                 lr=config['training']['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    scheduler_patch = StepLR(optimizer_patch, config['training']['decay_step'], gamma=config['training']['decay_gamma'])

    # loss
    loss_logic_fn = torch.nn.CrossEntropyLoss()
    mse_fn = torch.nn.MSELoss()
    
    # Best MRE record
    best_mre = 100.0
    best_epoch = -1
    alpha = config['training']['alpha'] = 0.99
    b = config['training']['batch_size']

    for epoch in range(config['training']['num_epochs']):
        net.train()
        net_patch.train()
        logic_loss_list = list()
        for index, (raw_img, crop_imgs, chosen_y, chosen_x, raw_y, raw_x) in enumerate(dataloader):
            with torch.autograd.set_detect_anomaly(False):
                raw_img = raw_img.cuda()
                crop_imgs = crop_imgs.cuda()

                raw_fea_list = net(raw_img)
                crop_fea_list = net_patch(crop_imgs)

                gt_y, gt_x = raw_y // (2 ** 5), raw_x // (2 ** 5)
                tmpl_y, tmpl_x = chosen_y // (2 ** 5), chosen_x // (2 ** 5)
                
                tmpl_feature = torch.stack([crop_fea_list[0][[id], :, tmpl_y[id], tmpl_x[id]] \
                                            for id in range(gt_y.shape[0])]).squeeze()
                ret_inner_5 = match_inner_product(raw_fea_list[0], tmpl_feature)  # shape [8,12,12]

                loss_5 = ce_loss(ret_inner_5, gt_y, gt_x)

                gt_y, gt_x = raw_y // (2 ** 4), raw_x // (2 ** 4)
                tmpl_y, tmpl_x = chosen_y // (2 ** 4), chosen_x // (2 ** 4)
                tmpl_feature = torch.stack([crop_fea_list[1][[id], :, tmpl_y[id], tmpl_x[id]] \
                                            for id in range(gt_y.shape[0])]).squeeze()
                ret_inner_4 = match_inner_product(raw_fea_list[1], tmpl_feature)
                loss_4 = ce_loss(ret_inner_4, gt_y, gt_x, nearby=config['training']['nearby'])

                gt_y, gt_x = raw_y // (2 ** 3), raw_x // (2 ** 3)
                tmpl_y, tmpl_x = chosen_y // (2 ** 3), chosen_x // (2 ** 3)
                tmpl_feature = torch.stack([crop_fea_list[2][[id], :, tmpl_y[id], tmpl_x[id]] \
                                            for id in range(gt_y.shape[0])]).squeeze()
                ret_inner_3 = match_inner_product(raw_fea_list[2], tmpl_feature)
                loss_3 = ce_loss(ret_inner_3, gt_y, gt_x, nearby=config['training']['nearby'])

                gt_y, gt_x = raw_y // (2 ** 2), raw_x // (2 ** 2)
                tmpl_y, tmpl_x = chosen_y // (2 ** 2), chosen_x // (2 ** 2)
                tmpl_feature = torch.stack([crop_fea_list[3][[id], :, tmpl_y[id], tmpl_x[id]] \
                                            for id in range(gt_y.shape[0])]).squeeze()
                ret_inner_2 = match_inner_product(raw_fea_list[3], tmpl_feature)
                loss_2 = ce_loss(ret_inner_2, gt_y, gt_x, nearby=config['training']['nearby'])

                gt_y, gt_x = raw_y // (2 ** 1), raw_x // (2 ** 1)
                tmpl_y, tmpl_x = chosen_y // (2 ** 1), chosen_x // (2 ** 1)
                tmpl_feature = torch.stack([crop_fea_list[4][[id], :, tmpl_y[id], tmpl_x[id]] \
                                            for id in range(gt_y.shape[0])]).squeeze()
                ret_inner_1 = match_inner_product(raw_fea_list[4], tmpl_feature)
                loss_1 = ce_loss(ret_inner_1, gt_y, gt_x, nearby=config['training']['nearby'])

                loss = loss_5 + loss_4 + loss_3 + loss_2 + loss_1

                optimizer.zero_grad()
                optimizer_patch.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_patch.step()

                logic_loss_list.append(np.array([loss_5.cpu().item(), loss_4.cpu().item(), \
                                                loss_3.cpu().item(), loss_2.cpu().item(), loss_1.cpu().item()]))
            if epoch == 0:
                print("Check code")
                break

        losses = np.stack(logic_loss_list).transpose()
        logger.info("Epoch {} Training logic loss 5 {:.3f} 4 {:.3f} 3 {:.3f} 2 {:.3f} 1 {:.3f}". \
                    format(epoch, losses[0].mean(), losses[1].mean(), losses[2].mean(), \
                           losses[3].mean(), losses[4].mean()))

        scheduler.step()
        scheduler_patch.step()

        if (epoch) % config['training']['save_seq'] == 0:
            net.eval()
            net_patch.eval()
            mre = tester.test(net, net_patch, dump_label=False)
            if mre < best_mre:
                best_mre = mre
                best_epoch = epoch
            logger.info(f"tag:{tag} ***********  Best MRE:{best_mre} in Epoch {best_epoch} || Epoch:{epoch}:{mre} ***********")
            logger.info(runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net_patch.state_dict(), runs_dir + "/model_patch_epoch_{}.pth".format(epoch))
            config['training']['last_epoch'] = epoch


    mre = tester.test(net, net_patch, dump_label=True, oneshot_id=126)
    dump_yaml(logger, config)

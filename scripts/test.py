import argparse
import datetime
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import os
import json
from PIL import Image, ImageDraw, ImageFont

from models.network import UNet_Pretrained
from datasets.ceph import Test_Cephalometric
from utils.eval import Evaluater
from utils.utils import to_Image, pred_landmarks, visualize, make_dir

from tutils import tfilename, trans_args, trans_init

def gray_to_PIL(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor  * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images

def gray_to_PIL2(tensor, pred_lm ,landmark, row=6, width=384):
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(images)
    red = (255, 0, 0)
    green = (0, 255, 0)
    # red = 255
    for i in range(row):
        draw.rectangle((pred_lm[0]+i*width-2, pred_lm[1]-2, pred_lm[0]+i*width+2, pred_lm[1]+2), fill=green)
        draw.rectangle((landmark[0]+i*width-2, landmark[1]-2, landmark[0]+i*width+2, landmark[1]+2), fill=red)
    draw.line([tuple(pred_lm), tuple(landmark)], fill='green', width=0)
    # import ipdb; ipdb.set_trace()
    return images

def match_cos(feature, template):
    feature = feature.permute(1, 2, 0)
    fea_L2 = torch.norm(feature, dim=-1)
    template_L2 = torch.norm(template, dim=-1)
    inner_product = (feature * template).sum(-1)
    cos_similarity = inner_product / (fea_L2 * template_L2 + 1e-3)
    return torch.clamp(cos_similarity, 0, 1)

def print_vars(vars:np.ndarray):
    for i, var in enumerate(vars):
        # print(var.shape)
        var = var / np.max(var) * 255
        image = Image.fromarray(var.astype(np.uint8))
        image.save(f"visuals/vars/var_{i}.jpg")
        import ipdb;ipdb.set_trace()

class Tester(object):
    def __init__(self, logger, config, net=None, tag=None, train="", args=None):
        mode = "Test1" if train == "" else "Train"

        dataset_1 = Test_Cephalometric(config['dataset']['pth'], 'Train')
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                shuffle=False, num_workers=config['training']['num_workers'])
        
        self.config = config
        self.args = args
        
        self.model = net 
        
        # Creat evluater to record results
        self.evaluater = Evaluater(logger, dataset_1.size, \
            dataset_1.original_size)

        self.logger = logger

        self.dataset = dataset_1
        self.id_landmarks = [i for i in range(config['training']['num_landmarks'])]

    def test(self, net, epoch=None, dump_label=False, tag=None, oneshot_id=126, draw=True):
        if net is not None:
            self.model = net
        assert(hasattr(self, 'model'))
        net.eval()
        config = self.config

        for i in range(1):
            one_shot_loader = Test_Cephalometric(pathDataset=self.config['dataset']['pth'], \
                mode='Oneshot', id_oneshot=oneshot_id)
            self.logger.info(f'ID Oneshot : {oneshot_id}')
            self.evaluater.reset()
            image, landmarks, template_patches, landmarks_ori = one_shot_loader.__getitem__(0)
            feature_list = list()
            image = image.cuda()
            features_tmp = net(image.unsqueeze(0))

            # Depth
            feature_list = dict()
            for id_depth in range(6):
                tmp = list()
                for id_mark, landmark in enumerate(landmarks_ori):
                    tmpl_y, tmpl_x = landmark[1] // (2 ** (5-id_depth)), landmark[0] // (2 ** (5-id_depth))
                    mark_feature = features_tmp[id_depth][0, :, tmpl_y, tmpl_x]
                    tmp.append(mark_feature.detach().squeeze().cpu().numpy())
                tmp = np.stack(tmp)
                one_shot_feature = torch.tensor(tmp).cuda()
                feature_list[id_depth] = one_shot_feature

            ID = 1

            for img, landmark_list in self.dataloader_1:
                # print("img shape", img.shape)
                img = img.cuda()
                features = net(img)
                
                pred_landmarks_y, pred_landmarks_x = list(), list()
                for id_mark in range(one_shot_feature.shape[0]):
                    cos_lists = []
                    cos_ori_lists = []
                    final_cos = torch.ones_like(img[0,0]).cuda()
                    for id_depth in range(5):
                        cos_similarity = match_cos(features[id_depth].squeeze(),\
                             feature_list[id_depth][id_mark])
                        # import ipdb;ipdb.set_trace()
                        cos_similarity = torch.nn.functional.upsample(\
                            cos_similarity.unsqueeze(0).unsqueeze(0), \
                            scale_factor=2**(5-id_depth), mode='nearest').squeeze()
                        # import ipdb;ipdb.set_trace()
                        final_cos = final_cos * cos_similarity
                        cos_lists.append(cos_similarity)
                        # import ipdb;ipdb.set_trace()
                    final_cos = (final_cos - final_cos.min()) / \
                        (final_cos.max() - final_cos.min())
                    cos_lists.append(final_cos)
                    chosen_landmark = final_cos.argmax().item()
                    pred_landmarks_y.append(chosen_landmark // 384)
                    pred_landmarks_x.append(chosen_landmark % 384)
                    debug = torch.cat(cos_lists, 1).cpu()
                    a_landmark = landmark_list[id_mark]
                    pred_landmark = (chosen_landmark % 384, chosen_landmark // 384)

                preds = [np.array(pred_landmarks_y), np.array(pred_landmarks_x)] 
                self.evaluater.record(preds, landmark_list)
                
                # Optional Save viusal results
                if draw:
                    image_pred = visualize(img, preds, landmark_list)
                    image_pred.save(tfilename(config['base']['runs_dir'], 'visuals', str(ID), 'pred.png'))

                if dump_label:
                    inference_marks = {id:[int(preds[1][id]), \
                        int(preds[0][id])] for id in range(19)}
                    dir_pth = tfilename(config['base']['runs_dir'], f"oneshot_id_126", 'pseudo-labels_init')
                    with open(tfilename('{0}/{1:03d}.json'.format(dir_pth, ID)), 'w') as f:
                        json.dump(inference_marks, f)
                    print("Dumped JSON file:" , '{0}/{1:03d}.json'.format(dir_pth, ID))
                
                ID += 1

            mre = self.evaluater.cal_metrics()
            return mre


if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='', help="name of the run")
    parser.add_argument("--config", default="cconf.yaml", help="default configs")
    parser.add_argument("--test", type=int, default=0, help="Test Mode")
    parser.add_argument("--resume", type=int , default=0)
    parser.add_argument("--pretrain", type=str, default="1") #emb-16-289
    parser.add_argument("--pepoch", type=int, default=0)
    parser.add_argument("--edge", type=int, default=1)
    parser.add_argument("--config", default="configs/config.yaml")
    args = trans_args(parser)
    logger, config = trans_init(args)
    tag = config['base']['tag']

    # Tester
    tester = Tester(logger, config, tag=args.tag)
    if args.edge > 0:
        config['edge_on'] = True
    else: config['edge_on'] = False

    net = UNet_Pretrained(3, non_local=config['training']['non_local'], emb_len=16, conf_m=True)
    net_patch = UNet_Pretrained(3, emb_len=16)

    if args.pretrain is not None and args.pretrain != "":
        epoch = args.pepoch
        print(f"Pretrain {tag} {epoch}")
        ckpt = "/data/quanquan/oneshot/runs-pixelpro/" + tag + f"/model_epoch_{epoch}.pth"
        assert os.path.exists(ckpt)
        print(f'Load CKPT {ckpt}')
        ckpt = torch.load(ckpt)
        net.load_state_dict(ckpt)
        ckpt2 = "/data/quanquan/oneshot/runs-pixelpro/" + tag + f"/model_patch_epoch_{epoch}.pth"
        assert os.path.exists(ckpt2)
        ckpt2 = torch.load(ckpt2)
        net_patch.load_state_dict(ckpt2)
    net = net.cuda()
    net_patch = net_patch.cuda()

    mre_list = []
    for i in range(150):
        mre = tester.test(net, net_patch, epoch=epoch, dump_label=False, oneshot_id=i+1, draw=False)
        mre_list.append(mre)
    np.save(f"mre_list_{tag}_{epoch}.npy", mre_list)
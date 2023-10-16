import numpy as np
import csv
import os
import json

import torch

from utils import make_dir, visualize, tensor_to_scaler


class Evaluater(object):
    def __init__(self, logger, num_landmark=19, eval_radius=[]):
        
        # self.tag = tag
        # make_dir(tag)
        # self.tag += '/'

        self.logger = logger

        self.RE_list = list()
        self.num_landmark = num_landmark

        self.recall_radius = eval_radius  # 2mm etc
        self.recall_rate = list()

        self.total_list = dict()

    def set_recall_radius(self, recall_radius):
        self.recall_radius = recall_radius

    def reset(self):
        self.RE_list.clear()
        self.total_list = list()

    def record(self, pred, landmark, scale_rate):
        # n = batchsize = 1
        # pred : list[ c(y) ; c(x) ]
        # landmark: list [ (y , x) * c]
        scale_rate_y = tensor_to_scaler(scale_rate[0])
        scale_rate_x = tensor_to_scaler(scale_rate[1])
        c = pred[0].shape[0]
        diff = np.zeros([c, 2], dtype=float)  # y, x
        for i in range(c):
            diff[i][0] = abs(pred[0][i] - landmark[i][0]) * scale_rate_y
            diff[i][1] = abs(pred[1][i] - landmark[i][1]) * scale_rate_x
        Radial_Error = np.sqrt(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2))
        self.RE_list.append(Radial_Error)
        # for i in range(len(Radial_Error)):
        #     if Radial_Error[i] > 10:
        #         print("Landmark {} RE {}".format(i, Radial_Error[i]))
        # if Radial_Error.max() > 10:
        #     return Radial_Error.argmax()
        return None
    
    def save_img(self, img, preds, landmark_list, runs_dir, id_str):
        
        self.path_visuals = os.path.join(runs_dir, 'visuals')
        if not os.path.isdir(self.path_visuals): make_dir(self.path_visuals)

        image_pred = visualize(img, preds, landmark_list)
        
        image_pred.save(os.path.join(self.path_visuals, f'{id_str}_pred.png'))

    def save_preds(self, preds, runs_dir, id_str):
        # Save format [y, x]
        inference_marks = {id:[int(preds[0][id]), \
            int(preds[1][id])] for id in range(self.num_landmark)}
        dir_pth = os.path.join(runs_dir, 'pseudo_labels')
        if not os.path.isdir(dir_pth): os.mkdir(dir_pth)
        with open('{0}/{1}.json'.format(dir_pth, id_str), 'w') as f:
            json.dump(inference_marks, f)

    def gen_latex(self):
        string_latex = f'{self.mre:.2f} & '
        for item in self.sdr:
            string_latex += f'{item:.2f} & '
        return string_latex

    def cal_metrics(self):
        # calculate MRE SDR
        temp = np.array(self.RE_list)
        Mean_RE_channel = temp.mean(axis=0)
        # self.logger.info(Mean_RE_channel)
        # with open('results.csv', 'w') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(Mean_RE_channel.tolist())
        self.logger.info("ALL MRE {}".format(Mean_RE_channel.mean()))
        self.mre = Mean_RE_channel.mean()

        self.sdr = []
        for radius in self.recall_radius:
            total = temp.size
            shot = (temp < radius).sum()
            self.logger.info("ALL SDR {}mm  {}".format\
                                 (radius, shot * 100 / total))
            self.sdr.append(shot * 100 / total)
        self.logger.info(self.gen_latex())

    
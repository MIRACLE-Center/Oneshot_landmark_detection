import numpy as np

from .utils import make_dir


class Evaluater(object):
    def __init__(self, logger, size, original_size, tag='paper_figure'):
        self.pixel_spaceing = 0.1
        self.tag = tag
        make_dir(tag)
        self.tag += '/'

        self.logger = logger
        self.scale_rate_y = original_size[0] / size[0]
        self.scale_rate_x = original_size[1] / size[1]

        self.RE_list = list()

        self.recall_radius = [2, 2.5, 3, 4]  # 2mm etc
        self.recall_rate = list()

        self.Attack_RE_list = list()
        self.Defend_RE_list = list()

        self.dict_Attack = dict()
        self.dict_Defend = dict()
        self.total_list = dict()

        self.mode_list = [0, 1, 2, 3]
        self.mode_dict = {0: "Iterative FGSM", 1: "Adaptive Iterative FGSM", \
                          2: "Adaptive_Rate", 3: "Proposed"}

        for mode in self.mode_list:
            self.dict_Defend[mode] = dict()
            self.dict_Attack[mode] = dict()
            self.total_list[mode] = list()
        self.best_mre = 100.0

    def reset(self):
        self.RE_list.clear()
        for mode in self.mode_list:
            self.dict_Defend[mode] = dict()
            self.dict_Attack[mode] = dict()
            self.total_list[mode] = list()
        self.Attack_RE_list.clear()
        self.Defend_RE_list.clear()

    def record(self, pred, landmark):
        # n = batchsize = 1
        # pred : list[ c(y) ; c(x) ]
        # landmark: list [ (x , y) * c]
        c = pred[0].shape[0]
        diff = np.zeros([c, 2], dtype=float)  # y, x
        for i in range(c):
            diff[i][0] = abs(pred[0][i] - landmark[i][1]) * self.scale_rate_y
            diff[i][1] = abs(pred[1][i] - landmark[i][0]) * self.scale_rate_x
        Radial_Error = np.sqrt(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2))
        Radial_Error *= self.pixel_spaceing
        self.RE_list.append(Radial_Error)
        # for i in range(len(Radial_Error)):
        #     if Radial_Error[i] > 10:
        #         print("Landmark {} RE {}".format(i, Radial_Error[i]))
        # if Radial_Error.max() > 10:
        #     return Radial_Error.argmax()
        return None

    def record_attack(self, pred, landmark, attack_list, mode=0, iteration=0):
        # n = batchsize = 1
        # pred : list[ c(y) ; c(x) ]
        # landmark: list [ (x , y) * c]
        assert (mode in [0, 1, 2, 3])

        c = pred[0].shape[0]
        diff = np.zeros([c, 2], dtype=float)  # y, x
        attack_temp = list()
        defend_temp = list()
        for i in range(c):
            diff[i][0] = abs(pred[0][i] - landmark[i][1]) * self.scale_rate_y
            diff[i][1] = abs(pred[1][i] - landmark[i][0]) * self.scale_rate_x
            Radial_Error = np.sqrt(np.power(diff[i, 0], 2) + np.power(diff[i, 1], 2))
            if i in attack_list:
                attack_temp.append([i, Radial_Error * self.pixel_spaceing])
            else:
                defend_temp.append([i, Radial_Error * self.pixel_spaceing])

        if iteration not in self.dict_Attack[mode].keys():
            self.dict_Attack[mode][iteration] = list()
        self.dict_Attack[mode][iteration].append(attack_temp)
        if iteration not in self.dict_Defend[mode].keys():
            self.dict_Defend[mode][iteration] = list()
        self.dict_Defend[mode][iteration].append(defend_temp)

    def cal_metrics(self, ex=False):
        # calculate MRE SDR
        temp = np.array(self.RE_list)
        Mean_RE_channel = temp.mean(axis=0)
        self.logger.info(Mean_RE_channel)
        # with open('./tmp/results.csv', 'w') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(Mean_RE_channel.tolist())
        mre = Mean_RE_channel.mean()
        self.logger.info("ALL MRE {}".format(mre))

        for radius in self.recall_radius:
            total = temp.size
            shot = (temp < radius).sum()
            self.logger.info("ALL SDR {}mm  {}".format\
                                 (radius, shot * 100 / total))
        if ex:
            return mre, None
        return mre

    
import numpy as np 
import torch
import time
from pathlib import Path

from multiprocessing import Process, Queue
from PIL import Image
from torchvision.transforms import ToPILImage

from PIL import Image, ImageDraw, ImageFont

to_PIL = ToPILImage()

def pred2gt(pred):
    if len(pred) != 2: 
        return pred
    # Convert predicts to GT format
    # pred :  list[ c(y) ; c(x) ]
    out = list()
    for i in range(pred[0].shape[-1]):
        out.append([int(pred[1][i]), int(pred[0][i])])
    return out

def distance(pred, landmark, k):
    diff = np.zeros([2], dtype=float) # y, x
    diff[0] = abs(pred[0] - landmark[k][1]) * 3.0
    diff[1] = abs(pred[1] - landmark[k][0]) * 3.0
    Radial_Error = np.sqrt(np.power(diff[0], 2) + np.power(diff[1], 2))
    Radial_Error *= 0.1
    # if Radial_Error > 40:
    #     return Radial_Error
    return 0

def to_Image(tensor, show=None, normalize=False):
    if normalize:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = tensor.cpu()
    image = to_PIL(tensor)
    if show:
        image.save(show + ".png")
    return image

def voting_channel(k, heatmap, regression_y, regression_x,\
     Radius, spots_y, spots_x, queue, num_candi):
    n, c, h, w = heatmap.shape

    score_map = np.zeros([h, w], dtype=int)
    for i in range(num_candi):
        vote_x = regression_x[0, k, spots_y[0, k, i], spots_x[0, k, i]]
        vote_y = regression_y[0, k, spots_y[0, k, i], spots_x[0, k, i]]
        vote_x = spots_x[0, k, i] + int(vote_x * Radius)
        vote_y = spots_y[0, k, i] + int(vote_y * Radius)
        if vote_x < 0 or vote_x >= w or vote_y < 0 or vote_y >= h:
            # Outbounds
            continue
        score_map[vote_y, vote_x] += 1
    score_map = score_map.reshape(-1)
    candidataces = score_map.argsort()[-10:]
    candidataces_x = candidataces % w
    candidataces_y = candidataces / w
    # import ipdb; ipdb.set_trace()
    # Print Big mistakes
    # gg = distance([candidataces_y[-1], candidataces_x[-1]], gt, k)
    # if gg:
    #     print("Landmark {} RE {}".format(k, gg))
    #     print(candidataces_y.astype(int))
    #     print(candidataces_x.astype(int))
    #     print(gt[k][1], gt[k][0])
    queue.put([k, score_map.argmax()])

def pred_landmarks(heatmap):
    # n = batchsize = 1
    heatmap = heatmap.cpu()
    n, c, h, w = heatmap.shape
    assert(n == 1)

    heatmap = heatmap.squeeze().view(c, -1).detach().numpy()
            
    landmark = np.zeros([c], dtype=int)
    for i in range(c):
        landmark[i] = heatmap[i].argmax(-1)
    landmark_y = landmark / w
    landmark_x = landmark % w
    return [landmark_y.astype(int), landmark_x]

def visualize(img, landmarks, red_marks, ratio=0.01, draw_line=True):
    """
    Here we denote:
    landmarks: green, preds
    red_marks: red,  GT
    """
    # img : tensor [1, 3, h, w]
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    h, w = img.shape[-2], img.shape[-1]
    Radius_Base = int(min(h, w) * ratio)
    img = (img - img.min()) / (img.max() - img.min())
    img = img.cpu()
    num_landmarks = len(pred2gt(landmarks))
    # Draw Landmark
    # Green [0, 1, 0] Red [1, 0, 0]
    Channel_R = {'Red': 1, 'Green': 0, 'Blue': 0}
    Channel_G = {'Red': 0, 'Green': 1, 'Blue': 0}
    Channel_B = {'Red': 0, 'Green': 0, 'Blue': 1}
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 0)

    landmarks = pred2gt(landmarks)

    image = to_PIL(img[0])
    draw = ImageDraw.Draw(image)
    for i, landmark in enumerate(landmarks):
        red_id = red_marks[i]
        Radius = Radius_Base
        draw.rectangle((red_id[0]-Radius, red_id[1]-Radius,\
            red_id[0]+Radius, red_id[1]+Radius), fill=red)
        draw.rectangle((landmark[0]-Radius, landmark[1]-Radius, \
            landmark[0]+Radius, landmark[1]+Radius), fill=green)
        if draw_line:
            draw.line([tuple(landmark), (red_id[0], red_id[1])], fill='green', width=0)
        # font = ImageFont.truetype('ARIAL.TTF', size=9)
        # draw.text((landmark[0]-Radius, landmark[1]-Radius), str(i %19 + 1), \
        #     fill=blue) #, font=font
        # draw.text((red_id[0]-Radius, red_id[1]-Radius), str(i %19 + 1), \
        #     fill=blue) # , font=font

    
    return image

def make_dir(pth):
    dir_pth = Path(pth)
    if not dir_pth.exists():
        dir_pth.mkdir()
    return pth

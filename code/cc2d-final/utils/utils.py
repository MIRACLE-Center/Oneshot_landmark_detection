import numpy as np 
import torch
import time
from pathlib import Path

from multiprocessing import Process, Queue
from PIL import Image
from torchvision.transforms import ToPILImage
import cv2
from PIL import Image, ImageDraw, ImageFont

to_PIL = ToPILImage()

def tensor_to_scaler(item):
    if type(item) == torch.Tensor:
        return item.item()
    return item

def radial(pt1, pt2, factor=[1, 1]):
    return sum(((i-j)*s)**2 for i, j,s  in zip(pt1, pt2, factor))**0.5

def gray_to_PIL(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor  * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images

def np_rgb_to_PIL(tensor):
    images = Image.fromarray(tensor.astype(np.uint8))
    return images

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
     Radius, spots_y, spots_x, queue, num_candi, analysis=False):
    n, c, h, w = heatmap.shape

    score_map = np.zeros([h, w], dtype=float)

    for i in range(num_candi):
        vote_x = regression_x[0, k, spots_y[0, k, i], spots_x[0, k, i]]
        vote_y = regression_y[0, k, spots_y[0, k, i], spots_x[0, k, i]]
        vote_x = spots_x[0, k, i] + int(vote_x * Radius)
        vote_y = spots_y[0, k, i] + int(vote_y * Radius)
        if vote_x < 0 or vote_x >= w or vote_y < 0 or vote_y >= h:
            # Outbounds
            continue
        # if heatmap[0, k, spots_y[0, k, i], spots_x[0, k, i]] > 0.1:
        # score_map[vote_y, vote_x] += heatmap[0, k, spots_y[0, k, i], spots_x[0, k, i]]
        score_map[vote_y, vote_x] += 1
    score_map = score_map.reshape(-1)
    # candidataces = score_map.argsort()[-10:]
    # candidataces_x = candidataces % w
    # candidataces_y = candidataces / w
    # import ipdb; ipdb.set_trace()
    # Print Big mistakes
    # gg = distance([candidataces_y[-1], candidataces_x[-1]], gt, k)
    # if gg:
    #     print("Landmark {} RE {}".format(k, gg))
    #     print(candidataces_y.astype(int))
    #     print(candidataces_x.astype(int))
    #     print(gt[k][1], gt[k][0])
    if analysis: queue.put([k, score_map.argmax(), score_map.max(), score_map.reshape([h, w])])
    else:
        queue.put([k, score_map.argmax(), score_map.max()])

def voting(heatmap, regression_y, regression_x, Radius, \
    get_voting=False, analysis=False, multi_process=False, infer_heatmap=False):
         
    # n = batchsize = 1
    heatmap = heatmap.cpu().detach().numpy()
    regression_x, regression_y = regression_x.cpu().numpy(), regression_y.cpu().numpy()
    n, c, h, w = heatmap.shape
    assert(n == 1)
    
    num_candi = int(3.14 * Radius * Radius) 

    # Collect top num_candi points
    # print("Get in done")
    spots = heatmap.reshape(n, c, -1).argsort(axis=-1)[:,:,-num_candi:]
    # print("ArgMax heatmap done") 
    spots_y = spots // w
    spots_x = spots % w    

    landmark = np.zeros([c], dtype=int)
    votings = np.zeros([c], dtype=int)
    score_map = np.zeros_like(heatmap[0])

    if not infer_heatmap:
        process_list = list()

        queue = Queue()
        if multi_process:
            for k in range(c):
                process = Process(target=voting_channel, args=(k, heatmap,\
                    regression_y, regression_x, Radius, spots_y, spots_x, queue, num_candi, analysis))
                process_list.append(process)
                process.start()
            for process in process_list:
                process.join()
        else:
            for k in range(c):
                voting_channel(k, heatmap, regression_y, regression_x, \
                    Radius, spots_y, spots_x, queue, num_candi, analysis)

        for i in range(c):
            out = queue.get()
            landmark[out[0]] = out[1]
            votings[out[0]] = out[2]
            if analysis:
                score_map[out[0]] = out[3] 
    else:
        # Heatmap 
        for i in range(c):
            landmark[i] = heatmap[0][i].reshape(-1).argmax()
    
    # print("Voting Done") 
    landmark_y = landmark / w
    landmark_x = landmark % w
    if analysis: return [landmark_y.astype(int), landmark_x], score_map
    if get_voting : return [landmark_y.astype(int), landmark_x], votings
    return landmark_y.astype(int), landmark_x

def heatmap_argmax(heatmap):
    n, c, h, w = heatmap.shape
    assert(n == 1)

    landmark = np.zeros([c], dtype=int)
    for i in range(c):
        landmark[i] = heatmap[0][i].reshape(-1).argmax()
    
    landmark_y = landmark / w
    landmark_x = landmark % w
    return landmark_y.astype(int), landmark_x


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

def visualize(img=torch.Tensor, pred_landmarks=None, gt_landmarks=None, ratio=0.005, draw_line=True):
    """
    # img : tensor [1, 3, h, w]
    pred_landmarks: [ [x,y], [x,y], ... ]  nd.ndarray, ground truth
    gt_landmarks: [ [x,y], [x,y], ... ]  nd.ndarray, predictions
    """

    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    h, w = img.shape[-2], img.shape[-1]
    Radius_Base = int(min(h, w) * ratio)
    img = (img - img.min()) / (img.max() - img.min())
    img = img.cpu()

    Channel_R = {'Red': 1, 'Green': 0, 'Blue': 0}
    Channel_G = {'Red': 0, 'Green': 1, 'Blue': 0}
    Channel_B = {'Red': 0, 'Green': 0, 'Blue': 1}
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 0)

    pred_landmarks = pred2gt(pred_landmarks)
    num = len(pred_landmarks)

    image = to_PIL(img[0])
    draw = ImageDraw.Draw(image)
    for i in range(num):
        Radius = Radius_Base
        
        if gt_landmarks is not None:
            gt_id = gt_landmarks[i][::-1]
            draw.rectangle((gt_id[0]-Radius, gt_id[1]-Radius, \
                            gt_id[0]+Radius, gt_id[1]+Radius), fill=green)
        if pred_landmarks is not None:
            pred_lm = pred_landmarks[i]
            draw.rectangle((pred_lm[0]-Radius, pred_lm[1]-Radius, \
                            pred_lm[0]+Radius, pred_lm[1]+Radius), fill=red)
        if draw_line and pred_landmarks is not None and gt_landmarks is not None:
            draw.line([(pred_lm[0], pred_lm[1]), (gt_id[0], gt_id[1])], fill='yellow', width=0)

    return image

def make_dir(pth):
    dir_pth = Path(pth)
    if not dir_pth.exists():
        dir_pth.mkdir()
    return pth

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = True,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    # import ipdb; ipdb.set_trace()
    if type(img == torch.tensor) : img = img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    if type(mask == torch.tensor) : mask = mask.detach().squeeze().cpu().numpy()

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def visual_gray_points(image, points):
    # image : gray array [h, w]
    # points [[x, y]]
    from imgaug.augmentables import KeypointsOnImage
    shape = image.shape
    kps = KeypointsOnImage.from_xy_array(np.array(points), image.shape)
    draw_image = kps.draw_on_image(np.stack([image, image, image], -1))
    return Image.fromarray(draw_image)

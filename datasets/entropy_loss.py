import torch
import torchvision
import numpy as np
import torch.nn as nn
import cv2

class MyEntropyLoss:
    """
    made for heatmap
    """
    def __init__(self, ref_landmarks=None, ref_image=None, with_ref=False):
        self.ref_landmarks = ref_landmarks
        self.num_classes = 19
        self.ref_image = ref_image
        self.with_ref = with_ref
        if ref_image is not None:
            self.ref_shape = ref_image.shape[:2]
        self.heatmaps = get_guassian_heatmaps_from_ref(self.ref_landmarks, self.num_classes, self.ref_shape)

    def entropy_loss(self, logits):
        if self.with_ref:
            return self.entropy_loss_with_ref(logits)
        else:
            # return self.normal_entropy_loss(logits)
            return self.entropy_based_regularization(logits)

    def normal_entropy_loss(self, logits):
        """
        Function:
            find a similar distribution
            calc the loss
        logits: heatmaps
        classes: number of classes
        """
        raise NotImplementedError
        # TODO: replaced with entropy_based_regularization()
        celoss = nn.CrossEntropyLoss()
        # points = logits.flatten()
        maps = logits
        min_loss = None
        for i in range(self.num_classes):
            loss = None
            for j, map in enumerate(maps):
                label = 1 if i == j else 0
                if loss is None:
                    loss = celoss(map, label)
                else:
                    loss += celoss(map, label)
            if min_loss is None or min_loss > loss:
                min_loss = loss

    def entropy_loss_with_ref(self, logits):
        """
        entropy loss with reference image heatmaps
        """
        # TODO:
        torch.argmax(logits)
        raise NotImplementedError
        return

    def entropy_based_regularization(self, logits, reduction=True):
        """
        logits.shape: (b, num_classes, m, n)
        return entropy: (b, m, n)
        """
        b, num_classes, m, n = logits.size()
        logits = logits.permute(b, m, n, num_classes)
        entropy = torch_entropy(logits)
        if reduction:
            entropy = torch.sum(entropy.view(b, -1))
        return entropy

    def get_weights_from_ref(self):
        """
        We should learn wights from the Reference Image
        ref_heatmap: heatmap of reference image

        """
        ref_maps = []
        for i, ref_map in enumerate(ref_maps):
            # find the areas we should focus, and consider the condition in other maps
            ref_map = np.where(np.repeat(self.heatmaps[i][np.newaxis, :, :], axis=0) > 0, self.heatmaps, 0)
            ref_maps.append(ref_map)
        ref_maps = np.stack(ref_maps, axis=0)  # shape: (19, 19, 800, 640)

        return ref_maps


def torch_entropy(p):
    """
    p.shape: (b,m,n, num_classes) => (b,m,n)
    """
    return torch.distributions.Categorical(probs=p).entropy()


def get_guassian_heatmaps_from_ref(landmarks, num_classes, image_shape=(800, 640), kernel_size=96, sharpness=0.2):
    """
    input: landmarks, [(1,1), ...]
    num_classes: number of classes
    image_shape: shape of the original image
    return:  shape: (19, 800, 640)
    """
    assert len(landmarks) == num_classes
    heatmaps = []
    for landmark in landmarks:
        # print("landmark: ", landmark)
        heatmaps.append(get_gaussian_heatmap_from_point(landmark, image_shape, size=kernel_size, sharpness=sharpness))
    return np.stack(heatmaps, axis=0)  # shape: (19, 800, 640)


def get_gaussian_heatmap_from_point(center_location, heatmap_shape=(96, 96), size=8, sharpness=0.2):
    """
    center_location: [x, y] the location the center of normal distribution
    heatmap_shape: the shape of output map, e.g. (96,96)
    size: size of heat area
    """
    # center_location[0], center_location[1] = int(center_location[0]), int(center_location[1])
    # assert type(center_location[0]) == int, "Expected int, bug got {}".format(type(center_location[0]))
    # assert type(center_location) == np.ndarray
    _, _, z = build_gaussian_layer(0, 1, len=size, sharpness=sharpness)
    # print("z.shape", z.shape)
    z = z - np.min(z)
    z_max = np.max(z)
    z /= z_max
    location_left = int(center_location[0])
    location_right = int(center_location[0] + size*2-1)
    location_top = int(center_location[1])
    location_bottom = int(center_location[1] + size*2-1)
    heatmap = np.zeros((heatmap_shape[0]+size*2, heatmap_shape[1]+size*2))
    # print(location_left, location_right)
    # import ipdb;ipdb.set_trace()
    heatmap[location_top:location_bottom, location_left:location_right] = z
    final_map = heatmap[size:-size, size:-size]
    assert final_map.shape[0] == heatmap_shape[0]
    return final_map

def build_gaussian_layer(mean, standard_deviation, step=1, len=8, sharpness=0.2):
    """
    copy from blog.csdn.net
    """
    # scaled_size = int(len / sharpness)
    scaled_size = len
    center_point = (scaled_size, scaled_size)
    x = np.arange(-scaled_size + 1, scaled_size, step) * (sharpness**2)
    y = np.arange(-scaled_size + 1, scaled_size, step) * (sharpness**2)
    x, y = np.meshgrid(x, y)
    z = np.exp(-((y - mean) ** 2 + (x - mean) ** 2) / (2 * (standard_deviation ** 2)))
    z = z / (np.sqrt(2 * np.pi) * standard_deviation)
    return (x, y, z)

def test_get_guassian_heatmaps_from_ref():
    a = get_guassian_heatmaps_from_ref([(100,100), (101,101)], num_classes=2, image_shape=(192, 192), kernel_size=96)
    a = a - np.min(a)
    a = a / np.max(a) * 255
    index = np.where(a[0]>0, 255, 0)
    print("a index", np.sum(index))
    aaa = a
    print(a[0].shape, np.max(a), np.min(a))
    print(a)
    import ipdb; ipdb.set_trace()
    cv2.imwrite("imgshow/gp-1.jpg", a[0].astype(np.uint8))
    cv2.imwrite("imgshow/gp-index-1.jpg", index.astype(np.uint8))

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # fig = plt.figure()
    # ax = Axes3D(fig)
    #
    # x3, y3, z3 = build_gaussian_layer(0, 1)
    # print("shapes: ", x3.shape)
    # import ipdb;
    # ipdb.set_trace()
    # ax.plot_surface(x3, y3, z3, rstride=1, cstride=1, cmap='rainbow')
    # plt.show()
    # plt.savefig("test_guassion.png")
    test_get_guassian_heatmaps_from_ref()
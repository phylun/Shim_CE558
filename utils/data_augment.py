import torch
from torchvision import transforms
import cv2
import numpy as np
import random
import math
from utils.box_utils import matrix_iou
from imgaug import augmenters as iaa
from model import config
C = config.Config()

def _crop(image, boxes, labels):
    height, width, _ = image.shape

    if len(boxes) == 0:
        return image, boxes, labels

    while True:
        mode = random.choice((
            None,
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ))

        if mode is None:
            return image, boxes, labels

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            scale = random.uniform(0.3, 1.)
            min_ratio = max(0.5, scale * scale)
            max_ratio = min(2, 1. / scale / scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            w = int(scale * ratio * width)
            h = int((scale / ratio) * height)

            l = random.randrange(width - w)
            t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            iou = matrix_iou(boxes, roi[np.newaxis])

            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                .all(axis=1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            if len(boxes_t) == 0:
                continue

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            return image_t, boxes_t, labels_t


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _expand(image, boxes, fill, p):
    if random.random() > p:
        return image, boxes

    height, width, depth = image.shape
    for _ in range(50):
        scale = random.uniform(1, 4)

        min_ratio = max(0.5, 1. / scale / scale)
        max_ratio = min(2, scale * scale)
        ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
        ws = scale * ratio
        hs = scale / ratio
        if ws < 1 or hs < 1:
            continue
        w = int(ws * width)
        h = int(hs * height)

        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        boxes_t = boxes.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:] += (left, top)

        expand_image = np.empty(
            (h, w, depth),
            dtype=image.dtype)
        expand_image[:, :] = fill
        expand_image[top:top + height, left:left + width] = image
        image = expand_image

        return image, boxes_t


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc_for_test(image, insize, mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= mean
    return image.transpose(2, 0, 1)


def _hori_mirror(image, boxes, clsimg_25, clsimg_50, clsimg_100, clsimg_200):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        clsimg_25 = clsimg_25[:, ::-1]
        clsimg_50 = clsimg_50[:, ::-1]
        clsimg_100 = clsimg_100[:, ::-1]
        clsimg_200 = clsimg_200[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes, clsimg_25, clsimg_50, clsimg_100, clsimg_200


def _vert_mirror(image, boxes, clsimg_25, clsimg_50, clsimg_100, clsimg_200):
    height, _, _ = image.shape
    if random.randrange(2):
        image = image[::-1, :]
        clsimg_25 = clsimg_25[::-1, :]
        clsimg_50 = clsimg_50[::-1, :]
        clsimg_100 = clsimg_100[::-1, :]
        clsimg_200 = clsimg_200[::-1, :]
        boxes = boxes.copy()
        boxes[:, 1::2] = height - boxes[:, 3::-2]
    return image, boxes, clsimg_25, clsimg_50, clsimg_100, clsimg_200


def _con_norm(image, boxes):
    list_parm = [0.5, 0.7, 1.3, 1.5]
    sel = np.arange(len(list_parm))
    np.random.shuffle(sel)

    contnormal = iaa.ContrastNormalization(list_parm[int(sel[0])])
    if random.randrange(len(list_parm) + 1) > 0:
        # print('contrast normalization')
        image = contnormal.augment_image(image)

    return image, boxes


def _blur(image, boxes):
    list_parm = [3, 5, 7, 9]
    sel = np.arange(len(list_parm))
    np.random.shuffle(sel)

    avblur = iaa.AverageBlur(list_parm[int(sel[0])])
    if random.randrange(len(list_parm) + 1) > 0:
        # print('blur')
        image = avblur.augment_image(image)

    return image, boxes


def _color(image, boxes):
    list_parm = [-45, -30, 30, 45]
    sel = np.arange(len(list_parm))
    np.random.shuffle(sel)

    added = iaa.Add(list_parm[int(sel[0])])
    if random.randrange(len(list_parm) + 1) > 0:
        # print('color')
        image = added.augment_image(image)

    return  image, boxes


class preproc(object):

    def __init__(self, resize, rgb_means, p):
        self.means = rgb_means
        self.resize = resize
        self.p = p

    def __call__(self, image, targets, bin_image):
        boxes = targets[:,:-1].copy()
        # print(boxes)
        labels = targets[:,-1].copy()
        # print(labels)
        if len(boxes) == 0:
            #boxes = np.empty((0, 4))
            # print('No Bounding Boxes are Detected - 1 !!')
            targets = np.zeros((1,5))
            # targets = np.array([0, 0, 512-1, 512 -1, 0])
            # targets = np.expand_dims(targets, axis=0)
            image = preproc_for_test(image, self.resize, self.means)
            # print(image.shape, targets)
            return torch.from_numpy(image), targets, bin_image

        # clsimg_25_o = clsimg_25.copy()
        # clsimg_50_o = clsimg_50.copy()
        # clsimg_100_o = clsimg_100.copy()
        # clsimg_200_o = clsimg_200.copy()
        bin_image_o = bin_image.copy()

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:,:-1]
        labels_o = targets_o[:,-1]
        boxes_o[:, 0::2] /= width_o
        boxes_o[:, 1::2] /= height_o
        labels_o = np.expand_dims(labels_o,1)
        targets_o = np.hstack((boxes_o,labels_o))

        # image_t, boxes, labels = _crop(image, boxes, labels)
        # image_t = _distort(image_t)
        # image_t, boxes = _expand(image_t, boxes, self.means, self.p)
        # image_t, boxes = _mirror(image_t, boxes)
        #
        image_t, boxes = _color(image, boxes)
        image_t, boxes = _blur(image_t, boxes)
        image_t, boxes = _con_norm(image_t, boxes)


        # image_t, boxes, clsimg_25_t, clsimg_50_t, clsimg_100_t, clsimg_200_t = _hori_mirror(image_t, boxes, clsimg_25,
        #                                                                                     clsimg_50, clsimg_100,
        #                                                                                     clsimg_200)
        # image_t, boxes, clsimg_25_t, clsimg_50_t, clsimg_100_t, clsimg_200_t = _vert_mirror(image_t, boxes, clsimg_25_t,
        #                                                                                     clsimg_50_t, clsimg_100_t,
        #                                                                                     clsimg_200_t)

        # clsimg_25_t = np.transpose(clsimg_25_t, [2, 0, 1])
        # clsimg_50_t = np.transpose(clsimg_50_t, [2, 0, 1])
        # clsimg_100_t = np.transpose(clsimg_100_t, [2, 0, 1])
        # clsimg_200_t = np.transpose(clsimg_200_t, [2, 0, 1])
        # print(image_t.shape, boxes.shape, clsimg_25_t.shape, clsimg_50_t.shape, clsimg_100_t.shape, clsimg_200_t.shape)
        # quit()

        # Mirroing
        # image_t, boxes = _mirror(image, boxes)



        height, width, _ = image_t.shape
        image_t = preproc_for_test(image_t, self.resize, self.means)
        boxes = boxes.copy()
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        b_w = (boxes[:, 2] - boxes[:, 0])*1.
        b_h = (boxes[:, 3] - boxes[:, 1])*1.
        mask_b= np.minimum(b_w, b_h) > 0.01
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()

        if len(boxes_t)==0:
            # print('No Bounding Boxes are Detected - 2 !!')
            image = preproc_for_test(image_o, self.resize, self.means)

            return torch.from_numpy(image), targets_o, bin_image_o

        labels_t = np.expand_dims(labels_t,1)
        targets_t = np.hstack((boxes_t,labels_t))

        # return torch.from_numpy(image_t), targets_t, clsimg_25_t, clsimg_50_t, clsimg_100_t, clsimg_200_t
        return torch.from_numpy(image_t), targets_t, bin_image_o


class BaseTransform(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """
    def __init__(self, resize, rgb_means, swap=(2, 0, 1)):
        self.means = rgb_means
        self.resize = resize
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img, t, b):

        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[0]
        img = cv2.resize(np.array(img), (self.resize,
                                         self.resize),interpolation = interp_method).astype(np.float32)
        img -= self.means
        img = img.transpose(self.swap)
        return torch.from_numpy(img), t, b
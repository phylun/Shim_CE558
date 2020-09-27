import numpy as np
from math import ceil
import math, os
from model import config
from utils.data_augment import preproc

import pickle
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
from utils.core import GetIndFromImg, GetPallete


if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

C = config.Config()


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)

class AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(C.CLASSES, range(len(C.CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0,5))
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                #cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res,bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class RoadDetection:
    def __init__(self, data_path, seg_file, preproc=None, target_transform=AnnotationTransform(), dataset_name='RoadDamage'):
        self.data_path = data_path
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join(data_path, '%s.xml')
        self._imgpath = os.path.join(data_path, '%s.jpg')
        self.ids = [f[:-4] for f in os.listdir(data_path) if f.find('.xml') > -1]
        self.fore_ids = list()
        self.back_ids = list()
        self.all_ids = list()
        # self.seg_data = np.load(seg_file, allow_pickle=True)
        self.seg_data = np.hstack((np.load(C.TEST_ROI_FILE, allow_pickle=True), np.load(C.TRAIN_ROI_FILE, allow_pickle=True)))
        self.split_data()
        self.pointer = 0

        self.batch_size = C.EVAL_IMS_PER_BATCH if dataset_name == 'test' else C.TRAIN_IMS_PER_BATCH

    def split_data(self):
        for i in range(self.__len__()):

            img, target, _ = self.__getitem__(i)
            # print(img.shape, target.shape)
            # if target[0, 2] == 0:
            if target.shape == (1, 5) and np.sum(target[0, :4]) == 0:
                # print('back_ids')
                self.back_ids += [i]
            else:
                self.fore_ids += [i]
            self.all_ids += [i]


    def __getitem__(self, index):

        file_name = self.ids[index]

        target = ET.parse(self._annopath % file_name).getroot()
        img = cv2.imread(self._imgpath % file_name)

        if self.target_transform is not None:
            target = self.target_transform(target)

        indx = GetIndFromImg(self.seg_data, file_name + '.jpg')
        indx = indx[0]
        # class_image = dict()
        # print(np.transpose(self.seg_data[indx]['class_image_25'], [2, 0, 1])
        binary_image = self.seg_data[indx]['binary_image'][:, :, 1]

        if self.preproc is not None:
            img, target, binary_image = self.preproc(img, target, binary_image)
            # img, target = self.preproc(img, target)

        return img, target, binary_image

    def pull_image(self, index):
        file_name = self.ids[index]
        return cv2.imread(self._imgpath % file_name, cv2.IMREAD_COLOR), file_name

    def pull_anno(self, index):
        file_name = self.ids[index]
        anno = ET.parse(self._annopath % file_name).getroot()
        gt = self.target_transform(anno)
        return gt


    def reset_pointer(self):
        permutation = np.random.permutation(self.forelen)
        self.fore_ids = [self.fore_ids[i] for i in permutation]

        permutation = np.random.permutation(self.backlen)
        self.back_ids = [self.back_ids[i] for i in permutation]

        permutation = np.random.permutation(self.alllen)
        self.all_ids = [self.all_ids[i] for i in permutation]

        self.pointer = 0


    def next_batch(self):
        bat_inputs = list()
        bat_targets = list()
        bat_binimage = list()

        for i in range(self.batch_size):
            # if i == self.backlen - 1:
            #     img, target = self.__getitem__(self.back_ids[int(np.random.randint(0, self.backlen))])
            # else:
            #     img, target = self.__getitem__(self.fore_ids[self.pointer + i])
            img, target, c_imgbin = self.__getitem__(self.all_ids[self.pointer + i])

            bat_inputs.append(img)
            bat_targets.append(target)
            bat_binimage.append(torch.from_numpy(c_imgbin.astype(np.float32)))

        bat_inputs, bat_targets = detection_collate((bat_inputs, bat_targets))
        self.pointer += self.batch_size

        return bat_inputs, bat_targets, torch.stack(bat_binimage, 0)


    def __len__(self):
        return len(self.ids)

    @property
    def forelen(self):
        return len(self.fore_ids)

    @property
    def forelen_batches_in_epoch(self):
        return int(math.floor(len(self.fore_ids) / self.batch_size))

    @property
    def backlen_batches_in_epoch(self):
        return int(math.floor(len(self.back_ids) / self.batch_size))

    @property
    def backlen(self):
        return len(self.back_ids)

    @property
    def alllen(self):
        return len(self.all_ids)

    @property
    def len_batches_in_epoch(self):
        return int(math.floor(len(self.all_ids) / self.batch_size))


if __name__ == '__main__':
    import cv2, os
    print('sb test scripts')
    data_path = '/home/javis/Data/bbox_200213_Test_Sample'

    seg_data = np.load(C.TEST_ROI_FILE, allow_pickle=True)

    _preproc = preproc(C.TRAIN_SCALES[0], C.RGB_MEAM, C.P)
    _AnnoTrans = AnnotationTransform()
    traindataloader = RoadDetection(data_path=data_path, seg_file=C.TEST_ROI_FILE, preproc=_preproc, target_transform=_AnnoTrans, dataset_name='train')

    empty_cnt = 0
    for ii in range(traindataloader.__len__()):
        # for ii in range(traindataloader.len_batches_in_epoch):

        # orig_img, target = traindataloader.next_batch()
        orig_img, target, bin_image = traindataloader.__getitem__(ii)
        file_name = traindataloader.ids[ii]

        re_p_classifier = np.zeros((C.TRAIN_SCALES[1], C.TRAIN_SCALES[0]))
        re_p_classifier = re_p_classifier/4 * 255
        seg_img = re_p_classifier.astype(np.uint8)

        # indx = GetIndFromImg(seg_data, file_name + '.jpg')
        # indx = indx[0]
        # seg_img = seg_data[indx]['binary_image'].astype(np.uint8)[:, :, 1]*255
        # print(seg_img.shape)

        # orig_img = np.array(batch_set[0])
        img = np.array(orig_img).copy()
        # target = batch_set[1]
        target = np.array(target)
        # quit()
        if target[0, 2] == 0:
            # print(target)
            print(target, empty_cnt)
            empty_cnt += 1
        # print(orig_img.shape)
        # print(target)
        # print(target)
        # quit()

        # img = orig_img.copy()
        img = np.transpose(img, (1, 2, 0))

        img += C.RGB_MEAM
        img = img.astype(np.uint8).copy()

        height_o, width_o, _ = img.shape
        # print(img.shape)
        for one_box in target:
            cls_name = C.CLASSES[int(one_box[-1])]

            cls_box = one_box[:4]
            cls_box[0::2] *= width_o
            cls_box[1::2] *= height_o
            cls_box = cls_box.astype(np.int)
            # print(cls_box)
            # print(cls_box[0], cls_box[1])

            cv2.rectangle(img, (cls_box[0], cls_box[1]), (cls_box[2], cls_box[3]), [255, 0, 0], thickness=2)
            cv2.putText(img, cls_name, (int(cls_box[0]+3), int(cls_box[1]+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 0], 1, cv2.LINE_AA)

        # for ii in range(1, 2):
        pal = GetPallete(1, seg_img, C.TRAIN_SCALES[0], C.TRAIN_SCALES[1])
        pal = pal.astype(np.uint8)
        img = np.add(img, (0.5 * pal))
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)

        cv2.imshow('Test', img)
        cv2.waitKey(0)

    print(empty_cnt)
    # batch = (list_img, list_tar)
    # print(detection_collate(batch))



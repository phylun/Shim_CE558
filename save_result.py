import numpy as np
from math import ceil
import math, os
from model import config
from utils.data_augment import preproc
from utils.core import *
import pickle
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
from db_loader import RoadDetection, AnnotationTransform
from utils.data_augment import preproc, BaseTransform
from model_deeplab.deeplab import DeepLab
from utils.core import GetIndFromImg, GetPallete
from eval_segm import *
import timeit
import torch.nn.functional as F


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


class TestRoadDetection:
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

            img, target, _, _= self.__getitem__(i)
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
        # clsimg_25 = torch.from_numpy(np.transpose(self.seg_data[indx]['class_image_25'], [2,0,1])[1])
        # clsimg_50 = torch.from_numpy(np.transpose(self.seg_data[indx]['class_image_50'], [2,0,1])[1])
        # clsimg_100 = torch.from_numpy(np.transpose(self.seg_data[indx]['class_image_100'], [2,0,1])[1])
        # clsimg_200 = torch.from_numpy(np.transpose(self.seg_data[indx]['class_image_200'], [2,0,1])[1])
        # clsimg_25 = self.seg_data[indx]['class_image_25'][:, :, 1]
        # clsimg_50 = self.seg_data[indx]['class_image_50'][:, :, 1]
        # clsimg_100 = self.seg_data[indx]['class_image_100'][:, :, 1]
        # clsimg_200 = self.seg_data[indx]['class_image_200'][:, :, 1]
        binary_image = self.seg_data[indx]['binary_image'][:, :, 1]

        if self.preproc is not None:
            img, target, binary_image = self.preproc(img, target, binary_image)
            # img, target = self.preproc(img, target)

        # return img, target, binary_image
        return img, target, binary_image, file_name

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
        # bat_cimg25 = list()
        # bat_cimg50 = list()
        # bat_cimg100 = list()
        # bat_cimg200 = list()
        bat_binimage = list()
        file_names = list()

        for i in range(self.batch_size):
            # if i == self.backlen - 1:
            #     img, target = self.__getitem__(self.back_ids[int(np.random.randint(0, self.backlen))])
            # else:
            #     img, target = self.__getitem__(self.fore_ids[self.pointer + i])
            # img, target, c_imgbin = self.__getitem__(self.all_ids[self.pointer + i])
            img, target, c_imgbin, file_name = self.__getitem__(self.all_ids[self.pointer + i])

            bat_inputs.append(img)
            bat_targets.append(target)
            # bat_cimg25.append(torch.from_numpy(c_img25.astype(np.float32)))
            # bat_cimg50.append(torch.from_numpy(c_img50.astype(np.float32)))
            # bat_cimg100.append(torch.from_numpy(c_img100.astype(np.float32)))
            # bat_cimg200.append(torch.from_numpy(c_img200.astype(np.float32)))
            bat_binimage.append(torch.from_numpy(c_imgbin.astype(np.float32)))

            file_names.append(file_name)

        bat_inputs, bat_targets = detection_collate((bat_inputs, bat_targets))
        self.pointer += self.batch_size

        # return bat_inputs, bat_targets, torch.stack(bat_binimage, 0)
        return bat_inputs, bat_targets, torch.stack(bat_binimage, 0), file_names


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




def main():
    """Create the model and start the evaluation process."""

    gpu0 = 0

    if not os.path.exists('Results'):
        os.makedirs('Results')

    # model = LinkNet(n_classes=2)
    # model = SegNet()
    # model = unet(n_classes=2)
    # model = DeepLab(backbone='mobilenet', output_stride=16)
    # model = DeepLab(backbone='resnet', output_stride=16)
    model = DeepLab(backbone='xception', output_stride=16)

    # saved_state_dict = torch.load('./outputs/CON_15000.pth')
    # model_names = [f for f in os.listdir('./output') if f.find('.pth') > -1]
    model_names = [f for f in os.listdir('./output') if f.find('h98.pth') > -1]
    # model_names = [f for f in os.listdir('./') if f.find('0.pth') > -1]
    score_list = list()
    for model_name in model_names:
        print(model_name)
        # saved_state_dict = torch.load('./outputs/FRRN_SEGMENTATION_epoch1000.pth')
        saved_state_dict = torch.load(os.path.join('output', model_name))
        # saved_state_dict = torch.load(os.path.join('./', model_name))
        model.load_state_dict(saved_state_dict)

        model.eval()
        model.cuda(gpu0)

        print_info('===> Loading Dataset...', ['yellow', 'bold'])
        _preproc = BaseTransform(C.TRAIN_SCALES[0], C.RGB_MEAM)
        _AnnoTrans = AnnotationTransform()

        testloader = TestRoadDetection(data_path=C.DATA_PATH_TEST, seg_file=C.TEST_ROI_FILE, preproc=_preproc,
                                   target_transform=_AnnoTrans, dataset_name='test')
        # testloader = Dataset(C.EVAL_IMS_PER_BATCH, C.TRAIN_ROI_FILE, C.INPUT_PATH)

        data_list = []

        eval_metrics = np.zeros(4)
        output_mat = list()
        gt_label_mat = list()

        testloader.pointer = 0
        total_time = 0
        for jj in range(testloader.forelen_batches_in_epoch):
            start_time = timeit.default_timer()

            with torch.no_grad():
                orig_images, orig_targets, bin_img, file_name = testloader.next_batch()
                # images = [imgs.float().cuda() for imgs in orig_images]
                images = orig_images.float().cuda(gpu0)
                targets = [anno.cuda(0) for anno in orig_targets]

                t_bin_img = bin_img.long().cuda(gpu0)

                out = model(images)
                # output = F.softmax(out, dim=1).cpu().detach()[0].numpy()
                output = out.cpu().detach()[0].numpy()
            end_time = timeit.default_timer()
            total_time += (end_time - start_time)

            # output = model(Variable(image, volatile=True).cuda(gpu0))
            # output = F.softmax(output, dim=1).cpu().detach()[0].numpy()
            # output = interp(output).cpu().detach().numpy()

            # output = output[:, :size[0],:size[1]]
            gt = np.asarray(t_bin_img.cpu().detach()[0].numpy()[:C.TRAIN_SCALES[0], :C.TRAIN_SCALES[1]], dtype=np.int)

            # print(output.shape)
            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

            # color_file = Image.fromarray(colorize(output).transpose(1, 2, 0), 'RGB')
            # color_file.save(filename)

            cimg = images.cpu().detach()[0].numpy()
            cimg = cimg.transpose(1, 2, 0)
            # cimg = cimg[:, :, ::-1]
            cimg += C.RGB_MEAM
            cimg = cimg.astype(np.uint8)

            gimg = cimg.copy()
            gt_label = t_bin_img.cpu().detach()[0].numpy()
            g_pal = GetPallete(3, gt_label * 255, C.TRAIN_SCALES[0], C.TRAIN_SCALES[1])
            pal = g_pal.astype(np.uint8)
            gimg = np.add(gimg, (0.5 * pal))
            gimg = np.clip(gimg, 0, 255).astype(np.uint8)
            gimg = gimg.astype(np.uint8)

            # Evaluation
            output_mat.append(output.flatten())
            gt_label_mat.append(gt_label.flatten())

            rimg = cimg.copy()
            # print(rimg.shape)
            o_pal = GetPallete(1, output * 255, C.TRAIN_SCALES[0], C.TRAIN_SCALES[1])
            pal = o_pal.astype(np.uint8)
            rimg = np.add(rimg, (0.5 * pal))
            rimg = np.clip(rimg, 0, 255).astype(np.uint8)

            # showimg = np.hstack((gimg, rimg))
            showimg = np.hstack((cimg, np.stack((g_pal[:, :, -1],) * 3, -1), np.stack((o_pal[:, :, 0],) * 3, -1))).astype(np.uint8)
            # showimg = np.hstack((cimg, gimg, rimg))
            cv2.imshow('Result', showimg)

            # quit()
            cv2.imwrite('./Results/' + file_name[0] + '_orig.jpg', cimg)
            cv2.imwrite('./Results/' + file_name[0] + '_true.jpg', np.stack((g_pal[:, :, -1],) * 3, -1))
            cv2.imwrite('./Results/' + file_name[0] + '_pred.jpg', np.stack((o_pal[:, :, 0],) * 3, -1))

            cv2.waitKey(1)
            # show_all(gt, output)
            # print(type(gt), gt.shape, gt)
            # print(type(output), output.shape, output)
            #
            # quit()
            data_list.append([gt.flatten(), output.flatten()])

        print('{:3.3f} ms'.format((total_time)*1000 / testloader.forelen_batches_in_epoch))

        filename = os.path.join('Results', 'result.txt')
        # print(model_name)
        output_mat = np.array(output_mat)
        gt_label_mat = np.array(gt_label_mat)
        eval_metrics[0] = pixel_accuracy(output_mat, gt_label_mat)
        eval_metrics[1] = mean_accuracy(output_mat, gt_label_mat)
        eval_metrics[2] = mean_IU(output_mat, gt_label_mat)
        eval_metrics[3] = frequency_weighted_IU(output_mat, gt_label_mat)


        print('pixel_accuracy: {:2.3f}%'.format(eval_metrics[0] * 100))
        print('mean_accuracy: {:2.3f}%'.format(eval_metrics[1] * 100))
        print('mean_IU: {:2.3f}%'.format(eval_metrics[2] * 100))
        print('frequency_weighted_IU: {:2.3f}%'.format(eval_metrics[3] * 100))
        score_list.append([str(eval_metrics[2]), model_name])

    arr_score = np.array(score_list)
    arr_score = list(map(float, arr_score[:, 0]))
    max_idx = np.argmax(arr_score)
    print(score_list[int(max_idx)])


if __name__ == '__main__':
    main()
    # print('SB test')
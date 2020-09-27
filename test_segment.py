import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
from collections import OrderedDict
from utils.core import *
from packaging import version
from eval_segm import *
import torch
import torch.nn as nn
from db_loader import RoadDetection, AnnotationTransform
from utils.data_augment import preproc, BaseTransform
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo

from model.linknet import LinkNet
from model.segnet import SegNet
from model.unetfromjin import unet
from model_deeplab.deeplab import DeepLab

from model import config
from eval_pre_rec_f1 import evaluate_segmentation
C = config.Config()
import timeit


from PIL import Image

def GetPallete(choice, cls_img, w, h):
    if choice == 1:
        b = np.float32(np.where(cls_img > 127, 255, 0))
        g = np.float32(np.zeros([w, h, 1]))
        r = np.float32(np.zeros([w, h, 1]))
    elif choice == 2:
        b = np.float32(np.zeros([w, h, 1]))
        g = np.float32(np.where(cls_img > 127, 255, 0))
        r = np.float32(np.zeros([w, h, 1]))
    elif choice == 3:
        b = np.float32(np.zeros([w, h, 1]))
        g = np.float32(np.zeros([w, h, 1]))
        r = np.float32(np.where(cls_img > 127, 255, 0))
    elif choice == 4:
        b = np.float32(np.where(cls_img > 127, 255, 0))
        g = np.float32(np.where(cls_img > 127, 255, 0))
        r = np.float32(np.where(cls_img > 127, 255, 0))
    else:
        b = np.zeros((h, w, 1), dtype=np.uint8)
        g = np.zeros((h, w, 1), dtype=np.uint8)
        r = np.zeros((h, w, 1), dtype=np.uint8)

    pallete = cv2.merge((b, g, r))

    return pallete




class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255
        # print(color_image.shape)
        # tmp = np.transpose(color_image, (1, 2, 0))
        # tmp = tmp.astype(np.uint8)
        # cv2.imshow('test', tmp)
        # cv2.waitKey(1)
        # quit()
        return color_image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    # classes = np.array(('background',  # always index 0
    #            'aeroplane', 'bicycle', 'bird', 'boat',
    #            'bottle', 'bus', 'car', 'cat', 'chair',
    #                      'cow', 'diningtable', 'dog', 'horse',
    #                      'motorbike', 'person', 'pottedplant',
    #                      'sheep', 'sofa', 'train', 'tvmonitor'))

    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.5f}'.format(i, C.SEG_CLASSES[i], j_list[i]))


    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, C.SEG_CLASSES[i], j_list[i]) + '\n')
            f.write('meanIOU: ' + str(aveJ) + '\n')

def show_all(gt, pred):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    classes = C.CLASSES
    colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5),
                    (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5),
                    (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0),
                    (0.5,0.75,0),(0,0.25,0.5)]
    cmap = colors.ListedColormap(colormap)
    bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('gt')
    ax1.imshow(gt, cmap=cmap, norm=norm)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    plt.show()

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


    # if args.pretrained_model != None:
    #     args.restore_from = pretrianed_models_dict[args.pretrained_model]
    #
    # if args.restore_from[:4] == 'http' :
    #     saved_state_dict = model_zoo.load_url(args.restore_from)
    # else:
    #     saved_state_dict = torch.load(args.restore_from)
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

        testloader = RoadDetection(data_path=C.DATA_PATH_TEST, seg_file=C.TEST_ROI_FILE, preproc=_preproc,
                                   target_transform=_AnnoTrans, dataset_name='test')
        # testloader = Dataset(C.EVAL_IMS_PER_BATCH, C.TRAIN_ROI_FILE, C.INPUT_PATH)

        data_list = []

        eval_metrics = np.zeros(4)
        output_mat = list()
        gt_label_mat = list()

        testloader.pointer = 0
        start_time = timeit.default_timer()
        for jj in range(testloader.len_batches_in_epoch):

            with torch.no_grad():

                orig_images, orig_targets, bin_img = testloader.next_batch()
                # images = [imgs.float().cuda() for imgs in orig_images]
                images = orig_images.float().cuda(1)
                targets = [anno.cuda(1) for anno in orig_targets]

                t_bin_img = bin_img.long().cuda(1)

                out = model(images)
                # check the final layer!!
                output = F.softmax(out, dim=1).cpu().detach()[0].numpy()
                # output = out.cpu().detach()[0].numpy()

            # output = model(Variable(image, volatile=True).cuda(gpu0))
            # output = F.softmax(output, dim=1).cpu().detach()[0].numpy()
            # output = interp(output).cpu().detach().numpy()

            # output = output[:, :size[0],:size[1]]
            gt = np.asarray(t_bin_img.cpu().detach()[0].numpy()[:C.TRAIN_SCALES[0],:C.TRAIN_SCALES[1]], dtype=np.int)

            # print(output.shape)
            output = output.transpose(1,2,0)
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
            g_pal = GetPallete(3, gt_label*255, C.TRAIN_SCALES[0], C.TRAIN_SCALES[1])
            pal = g_pal.astype(np.uint8)
            gimg = np.add(gimg, (0.5 * pal))
            gimg = np.clip(gimg, 0, 255).astype(np.uint8)
            gimg = gimg.astype(np.uint8)

            # Evaluation
            output_mat.append(output.flatten())
            gt_label_mat.append(gt_label.flatten())


            rimg = cimg.copy()
            # print(rimg.shape)
            o_pal = GetPallete(1, output*255, C.TRAIN_SCALES[0], C.TRAIN_SCALES[1])
            pal = o_pal.astype(np.uint8)
            rimg = np.add(rimg, (0.5 * pal))
            rimg = np.clip(rimg, 0, 255).astype(np.uint8)

            # showimg = np.hstack((gimg, rimg))
            showimg = np.hstack((cimg, np.stack((g_pal[:, :, -1],) * 3, -1), np.stack((o_pal[:, :, 0],) * 3, -1)))
            showimg = np.hstack((cimg, gimg, rimg))
            # cv2.imshow('Result', showimg)
            # cv2.waitKey(1)
            # quit()
            # cv2.imwrite(filename, showimg)
            # show_all(gt, output)
            # print(type(gt), gt.shape, gt)
            # print(type(output), output.shape, output)
            #
            # quit()
            data_list.append([gt.flatten(), output.flatten()])
        end_time = timeit.default_timer()
        print((end_time - start_time) / testloader.forelen_batches_in_epoch, 'Seconds')
        filename = os.path.join('Results', 'result.txt')

        output_mat = np.array(output_mat)
        gt_label_mat = np.array(gt_label_mat)
        eval_metrics[0] = pixel_accuracy(output_mat, gt_label_mat)
        eval_metrics[1] = mean_accuracy(output_mat, gt_label_mat)
        eval_metrics[2] = mean_IU(output_mat, gt_label_mat)
        eval_metrics[3] = frequency_weighted_IU(output_mat, gt_label_mat)
        get_iou(data_list, C.NUM_CLASSES, filename)

        re_val = evaluate_segmentation(output_mat, gt_label_mat, num_classes=2)
        print(re_val)

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
    # orig_data_list = np.load('tmp.npy', allow_pickle=True)
    # data_list = list()
    # for o_data in orig_data_list:
    #     data_list.append(o_data)
    #
    # # print(data_list[0][0].shape)
    # get_iou(data_list, C.NUM_CLASSES, 'original_results.txt')


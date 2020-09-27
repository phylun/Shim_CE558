#!/usr/bin/python

'''
Martin Kersner, m.kersner@gmail.com
2015/11/30

Evaluation metrics for image segmentation inspired by
paper Fully Convolutional Networks for Semantic Segmentation.
'''

import numpy as np
import os
import cv2


def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)

        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_


def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)

    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_


def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

    sum_k_t_k = get_pixel_area(eval_segm)

    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_


'''
Auxiliary functions used during evaluation.
'''


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def GetIndFromImg(roidb, filename):
    return [i for i, ele in enumerate(roidb) if ele['image_file'] == filename]


if __name__ == '__main__':

    y_true_img = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]])
    y_pred_img = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0]])
    print(pixel_accuracy(y_pred_img, y_true_img))
    print(mean_accuracy(y_pred_img, y_true_img))
    print(mean_IU(y_pred_img, y_true_img))
    print(frequency_weighted_IU(y_pred_img, y_true_img))



    # print('SB_Test')
    # input_path = 'D:\\MyProject\\Data\\RoadDamageDataset_v2\\RoadDamageDataset'
    # y_true = np.load(os.path.join(input_path, 'test_36_191028.npy'))
    # # y_pred = np.load('resize_y_pred.npy')
    # y_pred = np.load('ae_y_pred.npy')
    # # y_pred = np.load('origin_y_pred.npy')
    # # print(y_true[0]['class_image'].shape)
    # # quit()
    # eval_metrics = np.zeros(4)
    # for i, one_roi in enumerate(y_true):
    #     oidx = GetIndFromImg(y_pred, one_roi['image_file'])
    #     oidx = oidx[0]
    #     true_img = one_roi['binary_image'][:, :, 1]
    #     pred_img = np.where(y_pred[oidx]['y_pred'][:, :, 1] > 0.5, 255, 0)
    #     pred_img = pred_img.astype(np.uint8)
    #     res_img = np.hstack((true_img * 255, pred_img)).astype(np.uint8)
    #
    #
    #     # pred_img = np.where(y_pred[oidx]['y_pred'][0, :, :, 1] > 0.5, 255, 0)
    #     # pred_img = pred_img.astype(np.uint8)
    #     #
    #     # pred_img = cv2.resize(pred_img, (576, 576))
    #     # pred_img = cv2.threshold(pred_img, 127, 1, 0)[1]
    #     # res_img = np.hstack((true_img * 255, pred_img * 255)).astype(np.uint8)
    #     # cv2.imshow('test', tmp_img*255)
    #
    #     eval_metrics[0] += pixel_accuracy(true_img, pred_img)
    #     eval_metrics[1] += mean_accuracy(true_img, pred_img)
    #     eval_metrics[2] += mean_IU(true_img, pred_img)
    #     eval_metrics[3] += frequency_weighted_IU(true_img, pred_img)
    #     # acc += res
    #     # print(res)
    #
    #     cv2.imshow('test', res_img)
    #     cv2.waitKey(1)
    # print('pixel_accuracy: {:2.3f}%'.format(eval_metrics[0] * 100 / len(y_pred)))
    # print('mean_accuracy: {:2.3f}%'.format(eval_metrics[1] * 100 / len(y_pred)))
    # print('mean_IU: {:2.3f}%'.format(eval_metrics[2] * 100 / len(y_pred)))
    # print('frequency_weighted_IU: {:2.3f}%'.format(eval_metrics[3] * 100 / len(y_pred)))







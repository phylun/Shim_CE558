from termcolor import cprint
import os, shutil, time
import torch
from model import config
import numpy as np
import cv2


C = config.Config()



def GetIndFromImg(roidb, filename):
    return [i for i, ele in enumerate(roidb) if ele['image_file'] == filename]

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
    else:
        b = np.zeros((h, w, 1), dtype=np.uint8)
        g = np.zeros((h, w, 1), dtype=np.uint8)
        r = np.zeros((h, w, 1), dtype=np.uint8)

    pallete = cv2.merge((b, g, r))

    return pallete

def print_info(info, _type=None):
    if _type is not None:
        if isinstance(info,str):
            cprint(info, _type[0], attrs=[_type[1]])
        elif isinstance(info,list):
            for i in range(info):
                cprint(i, _type[0], attrs=[_type[1]])
    else:
        print(info)


def write_logger(info_dict,logger,iteration,status):
    if status:
        for tag,value in info_dict.items():
            logger.scalar_summary(tag, value, iteration)
    else:
        pass

def set_logger(status):
    if status:
        from logger import Logger
        date = time.strftime("%m_%d_%H_%M") + '_log'
        log_path = C.LOG_DIR + date
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        os.makedirs(log_path)
        logger = Logger(log_path)
        return logger
    else:
        pass


def print_train_log(iteration, print_epochs, info_list):
    if iteration % print_epochs == 0:
        # cprint('Time:{}||Epoch:{}||EpochIter:{}/{}||Iter:{}||Loss_L:{:.4f}||Loss_C:{:.4f}||Loss_CLS:{:.4f}||Batch_Time:{:.4f}'.format(*info_list), 'green')
        cprint('Time:{}||Epoch:{}||EpochIter:{}/{}||Iter:{}||Loss_Seg:{:.4f}||Batch_Time:{:.4f}'.format(*info_list), 'green')


def save_checkpoint(net, final=True, datasetname='RoadDamage',epoch=50):
    if final:
        torch.save(net.state_dict(), C.MODEL_OUT + \
                'Final_M2Det_{}_size{}_net{}.pth'.format(datasetname, C.TRAIN_SCALES[0], 'M2Det'))
    else:
        torch.save(net.state_dict(), C.MODEL_OUT + \
                'M2Det_{}_size{}_net{}_epoch{}.pth'.format(datasetname, C.TRAIN_SCALES, 'M2Det',epoch))


def image_forward(img, net, priors, detector, transform):
    w, h = img.shape[1], img.shape[0]
    scale = torch.Tensor([w, h, w, h])
    with torch.no_grad():
        x = transform(img).unsqueeze(0)
        x = x.cuda()
        scale = scale.cuda()

    out, p_cls = net(x)
    re_p_classifier = np.zeros((C.TRAIN_SCALES[1], C.TRAIN_SCALES[0], 2))
    for i in range(len(C.A_POOL)):
        tmp = p_cls[i].cpu().detach().numpy()[0, ...]
        tmp = np.transpose(tmp, (1, 2, 0))
        re_p_classifier += cv2.resize(tmp, (C.TRAIN_SCALES[1], C.TRAIN_SCALES[0]))
        # tmp =

    re_p_classifier /= 4

    boxes, scores = detector.forward(out, priors)
    boxes = (boxes[0] * scale).cpu().numpy()
    scores = scores[0].cpu().numpy()
    return boxes, scores, re_p_classifier


def nms_process(num_classes, i, scores, boxes, min_thresh, all_boxes, max_per_image):
    this_boxes = []
    for j in range(1, num_classes):  # ignore the bg(category_id=0)
        inds = np.where(scores[:, j] > min_thresh)[0]
        if len(inds) == 0:
            all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)

        # soft_nms = cfg.test_cfg.soft_nms
        # keep = nms(c_dets, cfg.test_cfg.iou, force_cpu=soft_nms)
        # keep = py_cpu_nms(c_dets, min_thresh)
        keep = non_max_suppression_fast(c_dets, min_thresh)
        # print(c_dets)
        # print(keep)
        # print(c_dets.shape)
        # quit()
        keep = keep[:C.KEEP_PER_CLASS]  # keep only the highest boxes
        c_dets = c_dets[keep, :]
        all_boxes[j][i] = c_dets

        this_boxes.extend([_.tolist()+[j] for _ in c_dets])

    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, num_classes):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j][i] = all_boxes[j][i][keep, :]

    return this_boxes


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        # inds = np.where(ovr >= thresh)[0]
        order = order[inds + 1]

    return keep



# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    # return boxes[pick].astype("int")
    return pick
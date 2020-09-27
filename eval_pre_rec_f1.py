import numpy as np
import os, cv2

def compute_avg_accuracy(y_pred, y_true):
    w = y_true.shape[0]
    h = y_true.shape[1]
    total = w*h
    count = 0.0
    for i in range(w):
        for j in range(h):
            if y_pred[i, j] == y_true[i, j]:
                count = count + 1.0
    return count / (total + 1e-8)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(y_pred, y_true, num_classes):
    w = y_true.shape[0]
    h = y_true.shape[1]
    flat_image = np.reshape(y_true, w*h)
    total = []
    for val in range(num_classes):
        total.append((flat_image == val).sum())

    count = [0.0] * num_classes
    for i in range(w):
        for j in range(h):
            if y_pred[i, j] == y_true[i, j]:
                count[int(y_pred[i, j])] = count[int(y_pred[i, j])] + 1.0

    # If there are no pixels from a certain class in the GT,
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / (total[i] + 1e-8))

    return accuracies


def precision(pred, label):
    TP = np.float(np.count_nonzero(pred * label))
    FP = np.float(np.count_nonzero(pred * (label - 1)))
    prec = TP / (TP + FP + 1e-8)
    return prec

# Compute recall
def recall(pred, label):
    TP = np.float(np.count_nonzero(pred * label))
    FN = np.float(np.count_nonzero((pred - 1) * label))
    rec = TP / (TP + FN + 1e-8)
    return rec

# Compute f1 score
def f1score(pred, label):
    prec = precision(pred, label)
    rec = recall(pred, label)
    f1 = np.divide(2 * prec * rec, (prec + rec + 1e-8))
    return f1


def compute_mean_iou(pred, label):
    w = label.shape[0]
    h = label.shape[1]
    unique_classes = np.unique(label)
    iou_list = list([0]) * len(unique_classes)

    for index, curr_class in enumerate(unique_classes):
        pred_mask = pred[:, :] == curr_class
        label_mask = label[:, :] == curr_class

        # TP = np.float(np.count_nonzero(pred_mask * label_mask))
        # FP = np.float(np.count_nonzero(pred_mask * (label_mask - 1)))
        # FN = np.float(np.count_nonzero((pred_mask - 1) * label_mask))
        iou_and = np.float(np.sum(np.logical_and(pred_mask, label_mask)))
        iou_or = np.float(np.sum(np.logical_or(pred_mask, label_mask)))
        iou_list[index] = iou_and / iou_or

    mean_iou = np.mean(iou_list)
    return mean_iou


def evaluate_segmentation(pred, gt, num_classes):
    accuracy = compute_avg_accuracy(pred, gt)
    class_accuracies = compute_class_accuracies(pred, gt, num_classes)
    prec = precision(pred, gt)
    rec = recall(pred, gt)
    f1 = f1score(pred, gt)
    iou = compute_mean_iou(pred, gt)
    # return accuracy, class_accuracies, prec, rec, f1, iou
    return accuracy, prec, rec, f1, iou


if __name__ == '__main__':
    label = np.array([[0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 1, 1]])
    pred = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 1, 1, 1, 1]])
    # print(evaluate_segmentation(pred, label, num_classes=4))
    # quit()
    data_dir = 'test_data'
    gt_dir = 'gt'
    gt_name_list = [os.path.join('.', data_dir, gt_dir, f) for f in os.listdir(os.path.join(data_dir, gt_dir)) if f.find('jpg') > -1]

    rs_dirs = ['rs_autob','rs_halfb'] # set the folder names of different methods

    eval_metrics = np.zeros((2, 5))
    for i, rs_dir in enumerate(rs_dirs):

        for gt_name in gt_name_list:
            img_name = gt_name.split('\\')[-1]
            gt_img = cv2.imread(gt_name, cv2.IMREAD_GRAYSCALE)
            gt_img = np.array(gt_img / 255.0).astype(np.int)

            rs_img = cv2.imread(os.path.join(data_dir, rs_dir, img_name), cv2.IMREAD_GRAYSCALE)
            rs_img = np.array(rs_img / 255.0).astype(np.int)
            re_val = np.array(evaluate_segmentation(rs_img, gt_img, num_classes=2))
            eval_metrics[i, :] += re_val


    eval_metrics /= len(gt_name_list)
    print(eval_metrics)




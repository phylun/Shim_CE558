import numpy as np

def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w*h


def calc_iou(a, b):
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def get_TP_map(true_boxes, detect_boxes, ref_iou, ref_prob, true_labels=None, detect_labels=None):

    T = []
    P = []

    bbox_match_flag = np.ones((true_boxes.shape[0]), np.int32)*-1
    if detect_boxes != []:
        detect_boxes = detect_boxes[detect_boxes[:, 0].argsort()[::-1]]
        d_boxes = detect_boxes[:, 1:5].astype(np.int32)
        d_prob = detect_boxes[:, 0]

        for i, d_box in enumerate(d_boxes):
            P.append(d_prob[i])
            found_match = False

            for j, t_box in enumerate(true_boxes):
                if bbox_match_flag[j] == 1:
                    continue

                iou_v = calc_iou(d_box, t_box)
                if iou_v > ref_iou and d_prob[i] > ref_prob:
                    found_match = True
                    bbox_match_flag[j] = 1
                    break
                else:
                    continue

            T.append(int(found_match))

    for i in range(true_boxes.shape[0]):
        if bbox_match_flag[i] == -1:
            T.append(1)
            P.append(0)

    return T, P
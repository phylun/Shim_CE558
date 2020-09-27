import cv2
from scipy import misc, ndimage
import numpy as np

def MyPerspective(src_img, r=50, strmode='constant'):
    mode = {
        "constant": cv2.BORDER_CONSTANT,
        "edge": cv2.BORDER_REPLICATE,
        "symmetric": cv2.BORDER_REFLECT,
        "reflect": cv2.BORDER_REFLECT_101,
        "wrap": cv2.BORDER_WRAP
    }

    h, w = src_img.shape
    # src_mat = np.array([[50,50], [50, 150], [150,50], [150, 150]], dtype=np.float32)
    # dst_mat = np.array([[70, 50],[30, 150], [130, 50], [170, 150]], dtype=np.float32)

    ratio = r
    src_mat = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    dst_mat = np.array([[ratio, ratio],[w-ratio-1, +ratio], [w+ratio+1, h-1-ratio], [-ratio, h-1-ratio]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_mat, dst_mat)
    wrapped = cv2.warpPerspective(src_img, M, (w, h), borderMode=mode[strmode])

    return wrapped


if __name__ == '__main__':
    ori_img = misc.imresize(ndimage.imread("./20170221_165156_pothole_00000or.jpg", mode='RGB')[0:200, 300:500, :], (200, 200))
    wrapped = MyPerspective(ori_img, r=80)
    cv2.imshow('original', ori_img)
    cv2.imshow('wrapped', wrapped)
    cv2.waitKey(0)

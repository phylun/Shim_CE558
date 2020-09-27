import numpy as np
import os

class Config:
    def __init__(self):
        self.CLASSES = ('__background', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'traffic_light', 'person')
        self.SEG_CLASSES = ('__background__', 'D00')
        self.CLS_TO_INT = dict(zip(self.CLASSES, range(len(self.CLASSES))))
        self.DATA_PATH_TRAIN = 'D:\\MyProject\\Data\\RoadDamageDataset_SB2\\bbox_200416_Train_old'
        self.DATA_PATH_TEST = 'D:\\MyProject\\Data\\RoadDamageDataset_SB2\\bbox_200416_Test_old'
        # self.DATA_PATH = 'D:\\MyProject\\Data\\RoadDamageDataset_SB2\\bbox_200113'

        SEG_PATH = 'D:\\MyProject\\Data\\RoadDamageDataset_SB2\\'

        # self.TRAIN_ROI_FILE = os.path.join(SEG_PATH, 'train_36_191101.npy')
        # self.TEST_ROI_FILE = os.path.join(SEG_PATH, 'test_36_191101.npy')

        self.TRAIN_ROI_FILE = os.path.join(SEG_PATH, 'train_200623_32.npy')
        self.TEST_ROI_FILE = os.path.join(SEG_PATH, 'test_200623_32.npy')

        self.PRE_TRAINED_FLAG = False
        self.TRANSFER_FLAG = False
        self.MODEL_OUT = './output/'
        self.LOG_DIR = '../logs/'

        self.ORIG_SCALES = (576, 576)
        self.TRAIN_SCALES = (512, 512)  # row & column
        # self.TRAIN_SCALES = (400, 400)  # row & column

        self.TEST_SCALES = (1024, 1024)

        self.TRAIN_IMS_PER_BATCH = 2
        self.EVAL_IMS_PER_BATCH = 1

        self.NUM_EPOCHS = 100

        # self.C_TYPE = ('36', '72', '144', '288')
        # self.A_POOL = ((36, 36), (72, 72), (144, 144), (288, 288))

        self.C_TYPE = ('25', '50', '100', '200')
        self.A_POOL = ((25, 25), (50, 50), (100, 100), (200, 200))

        self.Q_MONE = 'val_loss'  # quantity_monitor
        # self.Q_MONE = 'val_acc' # quantity_monitor
        self.M_MODE = 'auto'  # Monitor Mode
        self.SAVE_BEST = True

        self.PLANES = 256
        self.SMOOTH = True
        self.NUM_CLASSES = len(self.SEG_CLASSES)
        self.NUM_SCALES = 6
        self.NUM_LEVELS = 8
        self.BASE_OUT = [2, 4]

        self.STEP_PATTERN = [8, 16, 32, 64, 107, 320]
        self.SIZE_PATTERN = [0.06, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
        self.SFAM = False
        self.RGB_MEAM = (104, 117, 123)
        self.P = 0.6
        self.KEEP_PER_CLASS = 50
    # self.SAVE_BEST = False

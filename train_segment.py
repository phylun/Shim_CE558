import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from utils.nn_utils import *
from utils.core import *
from utils.data_augment import preproc
import matplotlib.pyplot as plt

from db_loader import RoadDetection, AnnotationTransform
from model import config
# from model import DenseNet
from torchsummary import summary
from utils.prior_box import PriorBox
from utils.loss import CrossEntropy2d
from model.linknet import LinkNet
from model.segnet import SegNet
from model.unetfromjin import unet
from model_deeplab.deeplab import DeepLab
import copy

C = config.Config()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print_info('----------------------------------------------------------------------\n'
           '|                       M2Det Training Program                       |\n'
           '----------------------------------------------------------------------',['yellow','bold'])

# logger = set_logger(status=True)
net = LinkNet(n_classes=2)
# net = SegNet()
# net = unet(n_classes=2)
# net = DeepLab(backbone='mobilenet', output_stride=16)
# net = DeepLab(backbone='resnet', output_stride=16)
# net = DeepLab(backbone='xception', output_stride=16)
bce_loss = CrossEntropy2d()
# cudnn.ben
# summary(net, (3, 512, 512))
optimizer = optim.Adam(net.parameters())

net.train()
net.cuda(0)
epoch = C.NUM_EPOCHS
print_info('===> Loading Dataset...',['yellow','bold'])

_preproc = preproc(C.TRAIN_SCALES[0], C.RGB_MEAM, C.P)
_AnnoTrans = AnnotationTransform()
traindataloader = RoadDetection(data_path=C.DATA_PATH_TRAIN, seg_file=C.TRAIN_ROI_FILE, preproc=_preproc, target_transform=_AnnoTrans, dataset_name='train')
# traindataloader = RoadDetection(data_path=C.DATA_PATH_TEST, preproc=_preproc, target_transform=_AnnoTrans, dataset_name='train')

print_info('===> Training M2Det on Road Damage Dataset', ['yellow','bold'])
# print(traindataloader.backlen)
# quit()
losses_history = np.zeros((0, 2))

for ii in range(C.NUM_EPOCHS):
    traindataloader.reset_pointer()

    # for jj in range(traindataloader.len_batches_in_epoch):
    avg_loss_seg = list()

    for jj in range(traindataloader.len_batches_in_epoch):
    # for jj in range(traindataloader.backlen_batches_in_epoch):
        iteration = jj+(ii*traindataloader.forelen_batches_in_epoch)
    #     iteration = jj+(ii*traindataloader.backlen_batches_in_epoch)
        load_t0 = time.time()
        orig_images, orig_targets, bin_img = traindataloader.next_batch()
        # images = [imgs.float().cuda() for imgs in orig_images]
        images = orig_images.float().cuda(0)
        targets = [anno.cuda(0) for anno in orig_targets]

        t_bin_img = bin_img.long().cuda(0)

        out = net(images)
        optimizer.zero_grad()

        loss_seg = bce_loss(out, t_bin_img)
        # loss = loss_c + loss_l + loss_cls
        avg_loss_seg.append(loss_seg.item())

        # write_logger({'seg_loss':loss_seg.item()},logger,iteration,status=True)

        loss_seg.backward()
        optimizer.step()
        load_t1 = time.time()
        print_train_log(iteration, 10,
                        # [time.ctime(), ii, ii%C.NUM_EPOCHS, C.NUM_EPOCHS, iteration, loss_l.item(),loss_c.item(), loss_cls.item() ,load_t1-load_t0])
                        [time.ctime(), ii, ii%C.NUM_EPOCHS, C.NUM_EPOCHS, iteration, loss_seg.item(),load_t1-load_t0])

    avg_loss_seg = np.array(avg_loss_seg)
    avg_loss_seg = avg_loss_seg.mean()

    tmp = np.array([ii, avg_loss_seg])
    tmp = np.expand_dims(tmp, axis=0)
    losses_history = np.vstack((losses_history, tmp))

    if (ii+1) % 1 == 0:
        save_checkpoint(net, final=False, epoch=ii+1)

np.save('./losses_history.npy', losses_history)

plt.figure('Losses History')
plt.plot(losses_history[:, 0], losses_history[:, 1], label='Seg')
# plt.plot(losses_history[:, 0], losses_history[:, 2], label='Cls')
plt.title('losses history along with iteration')
plt.legend(loc='best')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.grid(True)
# plt.show()
plt.draw()
plt.savefig('losses_history.png')
plt.close()

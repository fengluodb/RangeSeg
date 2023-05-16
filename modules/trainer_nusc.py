
#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import datetime
import os
import time
import imp
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from modules.trainer import Trainer
from dataset.nuscenes.parser import Parser
import __init__ as booger

import torch.optim as optim
from tensorboardX import SummaryWriter as Logger
from common.sync_batchnorm.batchnorm import convert_model
from modules.scheduler.warmupLR import warmupLR
from modules.scheduler.consine import CosineAnnealingWarmUpRestarts


from modules.loss.Lovasz_Softmax import Lovasz_softmax, Lovasz_softmax_PointCloud
from modules.loss.boundary_loss import BoundaryLoss
from modules.utils import AverageMeter, iouEval, save_checkpoint, show_scans_in_training, save_to_txtlog, make_log_img

class TrainerNusc(Trainer):
    def __init__(self, ARCH, DATA, datadir, logdir, path=None, point_refine=False):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.path = path
        self.epoch = 0
        self.point_refine = point_refine
        self.pipeline = self.ARCH["train"]["pipeline"]

        self.batch_time_t = AverageMeter()
        self.data_time_t = AverageMeter()
        self.batch_time_e = AverageMeter()

        # put logger where it belongs
        self.tb_logger = Logger(self.logdir + "/tb")
        self.info = {"train_update": 0,
                     "train_loss": 0, "train_acc": 0, "train_iou": 0,
                     "valid_loss": 0, "valid_acc": 0, "valid_iou": 0,
                     "best_train_iou": 0, "best_val_iou": 0}

        # get the data
        self.parser = Parser(root=self.datadir,
                             train_sequences=self.DATA["split"]["train"],
                             valid_sequences=self.DATA["split"]["valid"],
                             test_sequences=None,
                             split='train',
                             labels=self.DATA["labels"],
                             color_map=self.DATA["color_map"],
                             learning_map=self.DATA["learning_map"],
                             learning_map_inv=self.DATA["learning_map_inv"],
                             sensor=self.ARCH["dataset"]["sensor"],
                             max_points=self.ARCH["dataset"]["max_points"],
                             batch_size=self.ARCH["train"]["batch_size"],
                             workers=self.ARCH["train"]["workers"],
                             gt=True,
                             shuffle_train=True)


        self.set_loss_weight()
        self.set_model()
        self.set_gpu_cuda()
        self.set_loss_function(point_refine)
        self.set_optim_scheduler()

        # if need load the pre-trained model from checkpoint
        if self.path is not None:
            self.load_pretrained_model()

    def set_loss_weight(self):
        pass

    def set_loss_function(self, point_refine):
        """
            Used to define the loss function, multiple gpus need to be parallel
            # self.dice = DiceLoss().to(self.device)
            # self.dice = nn.DataParallel(self.dice).cuda()
        """
        # self.criterion = nn.NLLLoss(weight=self.loss_w).to(self.device)
        self.criterion = nn.NLLLoss(ignore_index=0).to(self.device)
        self.bd = BoundaryLoss().to(self.device)
        if not point_refine:
            self.ls = Lovasz_softmax(ignore=0).to(self.device)
        else:
            self.ls = Lovasz_softmax_PointCloud(ignore=0).to(self.device)

        # loss as dataparallel too (more images in batch)
        if self.n_gpus > 1:
            self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpus
            self.ls = nn.DataParallel(self.ls).cuda()
            self.bd = nn.DataParallel(self.bd).cuda()

    def init_evaluator(self):
        self.ignore_class = [0]
        for v in self.ignore_class:
            print("Ignoring class ", v, " in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                 self.device, self.ignore_class)
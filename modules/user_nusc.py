#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import os
import imp
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import __init__ as booger

from tqdm import tqdm
from modules.user import User
from dataset.nuscenes.parser import Parser
from utils.utils import *


class UserNusc(User):
    def __init__(self, ARCH, DATA, datadir, outputdir, modeldir, split, point_refine=False):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.outputdir = outputdir
        self.modeldir = modeldir
        self.split = split
        self.post = None
        self.infer_batch_size = 1
        self.point_refine = point_refine
        self.pipeline = self.ARCH["train"]["pipeline"]
        # get the data
        self.parser = Parser(root=self.datadir,
                             train_sequences=self.DATA["split"]["train"],
                             valid_sequences=self.DATA["split"]["valid"],
                             test_sequences=self.DATA["split"]["test"],
                             split=self.split,
                             labels=self.DATA["labels"],
                             color_map=self.DATA["color_map"],
                             learning_map=self.DATA["learning_map"],
                             learning_map_inv=self.DATA["learning_map_inv"],
                             sensor=self.ARCH["dataset"]["sensor"],
                             max_points=self.ARCH["dataset"]["max_points"],
                             batch_size=self.infer_batch_size,
                             workers=2,  # self.ARCH["train"]["workers"],
                             gt=True,
                             shuffle_train=False)

        with torch.no_grad():
            torch.nn.Module.dump_patches = True
            if not point_refine:
                self.set_model()
                checkpoint = self.pipeline + "_valid_best"
                w_dict = torch.load(
                    f"{self.modeldir}/{checkpoint}", map_location=lambda storage, loc: storage)
                try:
                    self.model = nn.DataParallel(self.model)
                    self.model.load_state_dict(
                        w_dict['state_dict'], strict=True)
                except:
                    self.set_model()
                    self.model.load_state_dict(
                        w_dict['state_dict'], strict=True)
                self.set_knn_post()
            else:
                from modules.PointRefine.spvcnn import SPVCNN
                self.set_model()
                self.model = nn.DataParallel(self.model)
                checkpoint = self.pipeline + "_refine_module_valid_best"
                w_dict = torch.load(
                    f"{self.modeldir}/{checkpoint}", map_location=lambda storage, loc: storage)
                # self.model.load_state_dict(w_dict['main_state_dict'], strict=True)
                self.model.load_state_dict(
                    {f"module.{k}": v for k, v in w_dict['main_state_dict'].items()}, strict=True)

                net_config = {'num_classes': self.parser.get_n_classes(),
                              'cr': 1.0, 'pres': 0.05, 'vres': 0.05}
                self.refine_module = SPVCNN(num_classes=net_config['num_classes'],
                                            cr=net_config['cr'],
                                            pres=net_config['pres'],
                                            vres=net_config['vres'])
                self.refine_module = nn.DataParallel(self.refine_module)
                w_dict = torch.load(
                    f"{modeldir}/{checkpoint}", map_location=lambda storage, loc: storage)
                # self.refine_module.load_state_dict(w_dict['state_dict'], strict=True)
                self.refine_module.load_state_dict(
                    {f"module.{k}": v for k, v in w_dict['refine_state_dict'].items()}, strict=True)

        self.set_gpu_cuda()

    def infer_subset(self, loader, to_orig_fn, cnn, knn):
        # switch to evaluate mode
        self.model.eval()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():

            end = time.time()

            for i, (proj_in, proj_mask, _, _, path_seq, path_name,
                    p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints, lidar_token)\
                    in enumerate(tqdm(loader, ncols=80)):
                # first cut to rela size (batch size one allows it)
                p_x = p_x[0, :npoints]
                p_y = p_y[0, :npoints]
                proj_range = proj_range[0, :npoints]
                unproj_range = unproj_range[0, :npoints]
                path_seq = path_seq[0]
                path_name = path_name[0]

                if self.gpu:
                    proj_in = proj_in.cuda()
                    p_x = p_x.cuda()
                    p_y = p_y.cuda()
                    if self.post:
                        proj_range = proj_range.cuda()
                        unproj_range = unproj_range.cuda()

                end = time.time()
                # compute output
                if self.ARCH["train"]["aux_loss"]["use"]:
                    proj_output, _, _, _ = self.model(proj_in)
                else:
                    proj_output, _ = self.model(proj_in)
                proj_argmax = proj_output[0].argmax(dim=0)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - end
                cnn.append(res)
                end = time.time()
                # print(f"Network seq {path_seq} scan {path_name} in {res} sec")

                # if knn --> use knn to postprocess
                # 	else put in original pointcloud using indexes
                if self.post:
                    unproj_argmax = self.post(proj_range, unproj_range,
                                              proj_argmax, p_x, p_y)
                else:
                    unproj_argmax = proj_argmax[p_y, p_x]

                # measure elapsed time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - end
                knn.append(res)
                # print(f"KNN Infered seq {path_seq} scan {path_name} in {res} sec")

                # save scan # get the first scan in batch and project scan
                pred_np = unproj_argmax.cpu().numpy()
                pred_np = pred_np.reshape((-1)).astype(np.uint8)

                # map to original label
                # pred_np = to_orig_fn(pred_np)

                if self.split == "test":
                    path = os.path.join(self.outputdir, "v1.0-test", "{}_lidarseg.bin".format(lidar_token[0]))
                else:
                    path = os.path.join(self.outputdir, "sequences",
                                        path_seq, "predictions", path_name)
                pred_np.tofile(path)

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
from dataset.poss.parser import Parser


class UserPoss(User):
    def __init__(self, ARCH, DATA, datadir, outputdir, modeldir, split, point_refine=False):
        super().__init__(ARCH, DATA, datadir, outputdir, modeldir, split, point_refine)

        self.parser = Parser(root=self.datadir,
                             train_sequences=self.DATA["split"]["train"],
                             valid_sequences=self.DATA["split"]["valid"],
                             test_sequences=None,
                             labels=self.DATA["labels"],
                             color_map=self.DATA["color_map"],
                             learning_map=self.DATA["learning_map"],
                             learning_map_inv=self.DATA["learning_map_inv"],
                             sensor=self.ARCH["dataset"]["sensor"],
                             max_points=self.ARCH["dataset"]["max_points"],
                             batch_size=self.infer_batch_size,
                             workers=2,
                             gt=True,
                             shuffle_train=False)

    def infer_subset(self, loader, to_orig_fn, cnn, knn):

        # switch to evaluate mode
        self.model.eval()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():

            end = time.time()

            proj_y = torch.full([40, 1800], 0, dtype=torch.long)
            proj_x = torch.full([40, 1800], 0, dtype=torch.long)
            for i in range(proj_y.size(0)):
                proj_y[i, :] = i
            for i in range(proj_x.size(1)):
                proj_x[:, i] = i

            proj_y = proj_y.reshape([40 * 1800])
            proj_x = proj_x.reshape([40 * 1800])
            proj_x = proj_x.cuda()
            proj_y = proj_y.cuda()

            for i, (proj_in, proj_labels, tags, unlabels, path_seq, path_name, proj_range, unresizerange, unproj_range, _, _)\
                    in enumerate(tqdm(loader, ncols=80)):
                # first cut to rela size (batch size one allows it)
                path_seq = path_seq[0]
                path_name = path_name[0]

                if self.gpu:
                    proj_in = proj_in.cuda()
                    unlabels = unlabels.cuda()
                    if self.post:
                        proj_range = proj_range[0].cuda()
                        unproj_range = unproj_range[0].cuda()

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
                                              proj_argmax, proj_x, proj_y)
                else:
                    unproj_argmax = proj_argmax[proj_y, proj_x]

                unproj_argmax = unproj_argmax[tags.squeeze()]
                # measure elapsed time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - end
                knn.append(res)
                # print(f"KNN Infered seq {path_seq} scan {path_name} in {res} sec")

                # save scan # get the first scan in batch and project scan
                pred_np = unproj_argmax.cpu().numpy()
                pred_np = pred_np.reshape((-1)).astype(np.int32)

                # map to original label
                pred_np = to_orig_fn(pred_np)

                path = os.path.join(self.outputdir, "sequences",
                                    path_seq, "predictions", path_name)
                pred_np.tofile(path)

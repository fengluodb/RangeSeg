
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

    def train_epoch(self, train_loader, model, criterion, optimizer,
                    epoch, evaluator, scheduler, color_fn, report=10,
                    show_scans=False):
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        hetero_l = AverageMeter()
        update_ratio_meter = AverageMeter()
        bd = AverageMeter()

        # empty the cache to train now
        # if self.gpu:
        #     torch.cuda.empty_cache()

        # switch to train mode
        model.train()

        end = time.time()
        for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name,
                _, _, _, _, _, _, _, _, _, _) in enumerate(train_loader):
            # measure data loading time
            self.data_time_t.update(time.time() - end)

            if not self.multi_gpu and self.gpu:
                in_vol = in_vol.cuda()
                #proj_mask = proj_mask.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda().long()

            if self.ARCH["train"]["aux_loss"]["use"]:
                [output, z2, z4, z8] = model(in_vol)
                lamda = self.ARCH["train"]["aux_loss"]["lamda"]
                bdlosss = self.bd(output, proj_labels.long()) + lamda[0]*self.bd(z2, proj_labels.long(
                )) + lamda[1]*self.bd(z4, proj_labels.long()) + lamda[2]*self.bd(z8, proj_labels.long())
                loss_m0 = criterion(torch.log(output.clamp(
                    min=1e-8)).double(), proj_labels).float() + 1.5 * self.ls(output, proj_labels.long())
                loss_m2 = criterion(torch.log(z2.clamp(
                    min=1e-8)).double(), proj_labels).float() + 1.5 * self.ls(z2, proj_labels.long())
                loss_m4 = criterion(torch.log(z4.clamp(
                    min=1e-8)).double(), proj_labels).float() + 1.5 * self.ls(z4, proj_labels.long())
                loss_m8 = criterion(torch.log(z8.clamp(
                    min=1e-8)).double(), proj_labels).float() + 1.5 * self.ls(z8, proj_labels.long())
                loss_m = loss_m0 + lamda[0]*loss_m2 + \
                    lamda[1]*loss_m4 + lamda[2]*loss_m8 + bdlosss
            else:
                output, _ = model(in_vol)
                bdlosss = self.bd(output, proj_labels.long())
                loss_m = criterion(torch.log(output.clamp(
                    min=1e-8)).double(), proj_labels).float() + self.ls(output, proj_labels.long()) + bdlosss

            optimizer.zero_grad()
            if self.n_gpus > 1:
                idx = torch.ones(self.n_gpus).cuda()
                loss_m.backward(idx)
                nn.utils.clip_grad.clip_grad_norm_(
                    self.model.parameters(), max_norm=1, norm_type=2)
            else:
                loss_m.backward()
                nn.utils.clip_grad.clip_grad_norm_(
                    self.model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()

            # measure accuracy and record loss
            loss = loss_m.mean()
            bd_loss = bdlosss.mean()
            with torch.no_grad():
                evaluator.reset()
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()

            losses.update(loss.item(), in_vol.size(0))
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))
            bd.update(bd_loss.item(), in_vol.size(0))

            # measure elapsed time
            self.batch_time_t.update(time.time() - end)
            end = time.time()

            # get gradient updates and weights, so I can print the relationship of
            # their norms
            update_ratios = []
            for g in self.optimizer.param_groups:
                lr = g["lr"]
                for value in g["params"]:
                    if value.grad is not None:
                        w = np.linalg.norm(
                            value.data.cpu().numpy().reshape((-1)))
                        update = np.linalg.norm(-max(lr, 1e-10)
                                                * value.grad.cpu().numpy().reshape((-1)))
                        update_ratios.append(update / max(w, 1e-10))
            update_ratios = np.array(update_ratios)
            update_mean = update_ratios.mean()
            update_std = update_ratios.std()
            update_ratio_meter.update(update_mean)  # over the epoch

            if show_scans:
                show_scans_in_training(
                    proj_mask, in_vol, argmax, proj_labels, color_fn)

            if i % self.ARCH["train"]["report_batch"] == 0:
                str_line = ('Lr: {lr:.3e} | '
                            'Update: {umean:.3e} mean,{ustd:.3e} std | '
                            'Epoch: [{0}][{1}/{2}] | '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                            'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                            'Bd {bd.val:.4f} ({bd.avg:.4f}) | '
                            'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                            'IoU {iou.val:.3f} ({iou.avg:.3f}) | [{estim}]').format(
                    epoch, i, len(train_loader), batch_time=self.batch_time_t,
                    data_time=self.data_time_t, loss=losses, bd=bd, acc=acc, iou=iou, lr=lr,
                    umean=update_mean, ustd=update_std, estim=self.calculate_estimate(epoch, i))
                print(str_line)
                save_to_txtlog(self.logdir, 'log.txt', str_line)

            # step scheduler
            scheduler.step()

        return acc.avg, iou.avg, losses.avg, update_ratio_meter.avg, hetero_l.avg

    def validate(self, val_loader, model, criterion, evaluator, class_func, color_fn, save_scans=False):
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        hetero_l = AverageMeter()
        rand_imgs = []

        # switch to evaluate mode
        model.eval()
        evaluator.reset()

        # empty the cache to infer in high res
        # if self.gpu:
        #     torch.cuda.empty_cache()

        with torch.no_grad():
            end = time.time()
            for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name,
                    _, _, _, _, _, _, _, _, _, _)\
                    in enumerate(tqdm(val_loader, desc="Validation:", ncols=80)):
                if not self.multi_gpu and self.gpu:
                    in_vol = in_vol.cuda()
                    proj_mask = proj_mask.cuda()
                if self.gpu:
                    proj_labels = proj_labels.cuda(non_blocking=True).long()

                # compute output
                if self.ARCH["train"]["aux_loss"]["use"]:
                    output, _, _, _ = model(in_vol)
                else:
                    output, _ = model(in_vol)
                log_out = torch.log(output.clamp(min=1e-8))

                # wce = criterion(log_out, proj_labels)
                jacc = self.ls(output, proj_labels)
                wce = criterion(log_out.double(), proj_labels).float()
                loss = wce + jacc

                # measure accuracy and record loss
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)

                losses.update(loss.mean().item(), in_vol.size(0))
                jaccs.update(jacc.mean().item(), in_vol.size(0))
                wces.update(wce.mean().item(), in_vol.size(0))

                if save_scans:
                    # get the first scan in batch and project points
                    mask_np = proj_mask[0].cpu().numpy()
                    depth_np = in_vol[0][0].cpu().numpy()
                    pred_np = argmax[0].cpu().numpy()
                    gt_np = proj_labels[0].cpu().numpy()
                    out = make_log_img(depth_np, mask_np,
                                       pred_np, gt_np, color_fn)
                    rand_imgs.append(out)

                # measure elapsed time
                self.batch_time_e.update(time.time() - end)
                end = time.time()

            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))

            str_line = ("*" * 80 + '\n'
                        'Validation set:\n'
                        'Time avg per batch {batch_time.avg:.3f}\n'
                        'Loss avg {loss.avg:.4f}\n'
                        'Jaccard avg {jac.avg:.4f}\n'
                        'WCE avg {wces.avg:.4f}\n'
                        'Acc avg {acc.avg:.6f}\n'
                        'IoU avg {iou.avg:.6f}').format(
                            batch_time=self.batch_time_e, loss=losses,
                            jac=jaccs, wces=wces, acc=acc, iou=iou)
            print(str_line)
            save_to_txtlog(self.logdir, 'log.txt', str_line)

            # print also classwise
            for i, jacc in enumerate(class_jaccard):
                self.info["valid_classes/" + class_func(i)] = jacc
                str_line = 'IoU class {i:} [{class_str:}] = {jacc:.6f}'.format(
                    i=i, class_str=class_func(i), jacc=jacc)
                print(str_line)
                save_to_txtlog(self.logdir, 'log.txt', str_line)
            str_line = '*' * 80
            print(str_line)
            save_to_txtlog(self.logdir, 'log.txt', str_line)

        return acc.avg, iou.avg, losses.avg, rand_imgs, hetero_l.avg
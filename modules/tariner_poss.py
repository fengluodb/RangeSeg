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
from dataset.poss.parser import Parser
import __init__ as booger


from modules.utils import AverageMeter, iouEval, save_checkpoint, show_scans_in_training, save_to_txtlog, make_log_img


class TrainerPoss(Trainer):
    def __init__(self, ARCH, DATA, datadir, logdir, path=None):
        super(TrainerPoss, self).__init__(ARCH, DATA, datadir, logdir, path, point_refine=False)

        # get the data
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
                             batch_size=self.ARCH["train"]["batch_size"],
                             workers=self.ARCH["train"]["workers"],
                             gt=True,
                             shuffle_train=True)

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
        for i, (in_vol, proj_labels, _, _, path_seq, path_name, _, _, _, _, _) in enumerate(train_loader):
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
                bdlosss = self.bd(output, proj_labels.long()) + lamda[0]*self.bd(z2, proj_labels.long()) + lamda[1]*self.bd(z4, proj_labels.long()) + lamda[2]*self.bd(z8, proj_labels.long())
                loss_m0 = criterion(torch.log(output.clamp(min=1e-8)).double(), proj_labels).float() + 1.5 * self.ls(output, proj_labels.long())
                loss_m2 = criterion(torch.log(z2.clamp(min=1e-8)).double(), proj_labels).float() + 1.5 * self.ls(z2, proj_labels.long())
                loss_m4 = criterion(torch.log(z4.clamp(min=1e-8)).double(), proj_labels).float() + 1.5 * self.ls(z4, proj_labels.long())
                loss_m8 = criterion(torch.log(z8.clamp(min=1e-8)).double(), proj_labels).float() + 1.5 * self.ls(z8, proj_labels.long())
                loss_m = loss_m0 + lamda[0]*loss_m2 + lamda[1]*loss_m4 + lamda[2]*loss_m8 + bdlosss
            else:
                output, _ = model(in_vol)
                bdlosss = self.bd(output, proj_labels.long())
                loss_m = criterion(torch.log(output.clamp(min=1e-8)).double(), proj_labels).float() + self.ls(output, proj_labels.long()) + bdlosss

            optimizer.zero_grad()
            if self.n_gpus > 1:
                idx = torch.ones(self.n_gpus).cuda()
                loss_m.backward(idx)
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
            else:
                loss_m.backward()
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
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
                        w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
                        update = np.linalg.norm(-max(lr, 1e-10) * value.grad.cpu().numpy().reshape((-1)))
                        update_ratios.append(update / max(w, 1e-10))
            update_ratios = np.array(update_ratios)
            update_mean = update_ratios.mean()
            update_std = update_ratios.std()
            update_ratio_meter.update(update_mean)  # over the epoch

            if show_scans:
                    # get the first scan in batch and project points
                    depth_np = in_vol[0][0].cpu().numpy()
                    pred_np = argmax[0].cpu().numpy()
                    gt_np = proj_labels[0].cpu().numpy()
                    out = self.make_log_img(depth_np, pred_np, gt_np, color_fn)

                    directory = os.path.join(self.log, "train-predictions")
                    if not os.path.isdir(directory):
                        os.makedirs(directory)
                    name = os.path.join(directory, str(i) + ".png")
                    cv2.imwrite(name, out)

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
            for i, (in_vol, proj_labels, _, _, path_seq, path_name, _, _, _, _, _)\
                    in enumerate(tqdm(val_loader, desc="Validation:", ncols=80)):
                if not self.multi_gpu and self.gpu:
                    in_vol = in_vol.cuda()
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
                jaccs.update(jacc.mean().item(),in_vol.size(0))
                wces.update(wce.mean().item(),in_vol.size(0))

                if save_scans:
                    # get the first scan in batch and project points
                    depth_np = in_vol[0][0].cpu().numpy()
                    pred_np = argmax[0].cpu().numpy()
                    gt_np = proj_labels[0].cpu().numpy()
                    out = Trainer.make_log_img(depth_np,
                                               pred_np,
                                               gt_np,
                                               color_fn)

                    directory = os.path.join(self.log, "valid-predictions")
                    if not os.path.isdir(directory):
                        os.makedirs(directory)
                    name = os.path.join(directory, str(i) + ".png")
                    cv2.imwrite(name, out)

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

    def make_log_img(depth, pred, gt, color_fn):
        # input should be [depth, pred, gt]
        # make range image (normalized to 0,1 for saving)
        depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        out_img = cv2.applyColorMap(
            depth, Trainer.get_mpl_colormap('viridis'))
        # make label prediction
        pred_color = color_fn(pred.astype(np.int32))
        out_img = np.concatenate([out_img, pred_color], axis=0)
        # make label gt
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
        return (out_img).astype(np.uint8)
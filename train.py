#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/17 下午2:22
# @Author  : jyl
# @File    : train.py
from train import Yolov3COCOTrainer
import torch
from dataset import COCODataset
from configs import opt
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from utils import plot_one
from torch.utils.data import DataLoader
import numpy as np
from utils import each_class_nms
from utils import mAP
from torchvision import transforms
from utils import parse_anchors
from utils import load_weights
from utils import letter_resize


class Yolov3:

    def __init__(self):
        self.VocTrainDataLoader = DataLoader(COCODataset(is_train=True), batch_size=opt.batch_size,
                                             shuffle=True, num_workers=opt.num_workers, drop_last=True, pin_memory=True)
        self.VocTestDataLoader = DataLoader(COCODataset(is_train=False), batch_size=1,
                                            shuffle=False, num_workers=0, drop_last=True, pin_memory=True)
        self.img_size = opt.img_size
        self.anchors = parse_anchors(opt.anchors_path)
        self.train_data_length = len(COCODataset(is_train=True))
        self.val_data_length = len(COCODataset(is_train=False))
        self.trainer = Yolov3COCOTrainer()
        self.testDataset = COCODataset(is_train=False)
        self.test_imgs = np.random.randint(0, self.val_data_length, 10)
        # self.test_imgs = np.random.randint(low=0, high=len(self.testDataset), size=35)
        self.normailze = transforms.Compose([transforms.ToTensor(), transforms.Normalize(opt.mean, opt.std)])
        self.logger = opt.logger
        self.writer = SummaryWriter(log_dir=os.path.join(opt.base_path, 'logs', f'summary_{opt.optimizer}'))

    def train(self):
        # self.writer.add_graph(self.trainer.yolov2.cpu(),
        # input_to_model=torch.rand(opt.batch_size, 3, opt.img_h, opt.img_w, device='cpu'))
        self.trainer.darknet53.train()
        loss_tmp = float('inf')
        if os.path.exists(opt.saved_model_path):
            self.logger.info(f'Use pretrained model: {opt.saved_model_path}')
            self.trainer.use_pretrain(opt.saved_model_path)
            loss_tmp = self.trainer.last_loss
            start_epoch = self.trainer.epoch_num
            total_steps = self.trainer.total_steps
        else:
            self.trainer.use_pretrain('/home/dk/jyl/V3/ckpt/darknet53.pkl')
            start_epoch = 0
            total_steps = 0
            self.logger.info('Train from "darknet53.pkl"')

        for epoch in range(start_epoch, opt.epochs):
            for i, data in tqdm(enumerate(self.VocTrainDataLoader)):
                img_rgb, target_13, target_26, target_52 = data
                img_rgb = img_rgb.to(opt.device)
                target_13 = target_13.to(opt.device)
                target_26 = target_26.to(opt.device)
                target_52 = target_52.to(opt.device)
                targets = [target_13, target_26, target_52]
                self.trainer.train_step(img_rgb, targets, epoch + 1)
                total_steps += 1
                mean_loss = self.trainer.meter.mean
                loss_dict = self.trainer.loss_dict
                self.add_train_summary(loss_dict, total_steps)
                if total_steps % opt.display_step == 0:
                    message = f'Epoch[{epoch: 03}] Step[{(epoch * self.train_data_length + i + 1): 06}]] \n' \
                              f'mean_loss : {mean_loss:.3f} \n' \
                              f'xy_loss   : {loss_dict["xy_loss"]:.3f} \n' \
                              f'wh_loss   : {loss_dict["wh_loss"]:.3f} \n' \
                              f'obj_conf_loss : {loss_dict["obj_conf_loss"]:.3f} \n' \
                              f'class_loss: {loss_dict["class_loss"]:.3f} \n' \
                              f'noobj_conf_loss  : {loss_dict["noobj_conf_loss"]:.3f} \n' \
                              f'total_loss: {loss_dict["total_loss"]:.3f} \n' \
                              f'learning rate: {self.trainer.scheduler.get_lr()[0]}'
                    self.logger.info(message)
                if total_steps % opt.eval_step == 0:
                    self.eval(epoch, i + 1)
                if total_steps % opt.save_step == 0:
                    if mean_loss < loss_tmp:
                        loss_tmp = mean_loss
                        self.trainer.save_model(epoch, total_steps, mean_loss, opt.model_save_dir + f'/model_best_{opt.optimizer}.pkl')

    def eval(self, epoch, step):
        conf_list, cls_list, score_list = [], [], []
        steps = 0
        with torch.no_grad():
            for img_id in self.test_imgs:
                steps += 1
                ori_img, resized_img, true_box, true_label = self.testDataset[img_id]
                _, ratio, dh, dw = letter_resize(ori_img, [416, 416])
                # img = self.normailze(resized_img)
                img = torch.from_numpy(resized_img / 255.0).float()
                img = img.permute(2, 0, 1)
                # [3, 416, 416] -> [1, 3, 416, 416]
                img = torch.unsqueeze(img, dim=0)
                img = img.to(opt.device)
                # boxes: [1, (13*13+26*26+52*52)*3, 4] / [xmin, ymin, xmax, ymax]
                # confs: [1, (13*13+26*26+52*52)*3, 1]
                # probs: [1, (13*13+26*26+52*52)*3, 80]
                boxes, confs, probs = self.trainer.predict(img)

                boxes = boxes.squeeze(0)  # [13*13*5, 4]
                confs = confs.squeeze(0)  # [13*13*5, 1]
                probs = probs.squeeze(0)  # [13*13*5, 80]
                scores = confs * probs  # [13*13*5, 80]

                conf_list.extend(confs.detach().cpu().numpy())
                cls_list.extend(probs.detach().cpu().numpy())
                score_list.extend(scores.detach().cpu().numpy())
                pred_dict = {'conf': conf_list[-1].flatten(), 'cls': cls_list[-1].flatten(), 'score': score_list[-1].flatten()}
                self.add_test_summary(pred_dict, steps)
                # box_out: [xmin, ymin, xmax, ymax]
                box_out, score_out, label_out = each_class_nms(boxes, scores, opt.score_threshold, opt.iou_threshold,
                                                               opt.max_boxes_num, self.img_size)

                if len(box_out) != 0:
                    plot_dict = {'img': ori_img,
                                 'ratio': ratio,
                                 'dh': dh,
                                 'dw': dw,
                                 'pred_box': box_out.cpu().numpy(),
                                 'pred_score': score_out.cpu().numpy(),
                                 'pred_label': label_out.cpu().numpy(),
                                 'gt_box': true_box,
                                 'gt_label': true_label,
                                 'img_name': f'epoch_{epoch}_step{step}_{img_id}.jpg',
                                 'save_path': os.path.join(opt.base_path, 'data', 'result')}

                    plot_one(plot_dict)

        msg = f"\n" \
              f"Score Mean: {np.mean(score_list):.5f} \n" \
              f"Score Max: {np.max(score_list):.5f} \n" \
              f"Score Min: {np.min(score_list):.5f} \n" \
              f"Max conf: {np.max(conf_list)} \n" \
              f"Max cls: {np.max(cls_list)}"
        self.logger.info(msg)

        self.trainer.darknet53.train()

    def test(self):
        mAP_predicts = []
        mAP_ground_truths = []
        self.trainer.use_pretrain(opt.saved_model_path)
        torch_normailze = transforms.Compose([transforms.ToTensor(), transforms.Normalize(opt.mean, opt.std)])
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.VocTestDataLoader)):
                ori_img, resized_img, true_boxes, true_labels = data
                img = torch_normailze(resized_img[0].numpy())
                img = img.unsqueeze(dim=0).to(opt.device)

                # boxes shape: [1, (13*13+26*26+52*52)*3, 4]
                # confs shape: [1, (13*13+26*26+52*52)*3, 1]
                # probs shape: [1, (13*13+26*26+52*52)*3, 80]
                boxes, confs, probs = self.trainer.predict(img)
                boxes = boxes.squeeze(0)  # [13*13*5, 4]
                confs = confs.squeeze(0)  # [13*13*5, 1]
                probs = probs.squeeze(0)  # [13*13*5, 80]
                scores = confs * probs  # [13*13*5, 80]

                # NMS
                box_output, score_output, label_output = each_class_nms(boxes, scores, opt.score_threshold, opt.iou_threshold,
                                                                        opt.max_boxes_num, self.img_size)

                _, ratio, dh, dw = letter_resize(ori_img.numpy()[0], [416, 416])
                if len(box_output) != 0:
                    plot_dict = {'img': ori_img.cpu().numpy()[0],
                                 'ratio': ratio,
                                 'dh': dh,
                                 'dw': dw,
                                 'pred_box': box_output.cpu().numpy(),
                                 'pred_score': score_output.cpu().numpy(),
                                 'pred_label': label_output.cpu().numpy(),
                                 'gt_box': None,
                                 'gt_label': None,
                                 'img_name': f'{i}.jpg',
                                 'save_path': '/home/dk/Desktop/model_best/'}
                    plot_one(plot_dict)
                if box_output.size(0) != 0:
                    # [X, 5]
                    mAP_predict_in = torch.cat([box_output, score_output.reshape(score_output.numel(), 1)], dim=-1)
                else:
                    mAP_predict_in = torch.zeros(1, 5)

                # shape: [N, 4]; format: [xmin, ymin, xmax, ymax]; dtype: np.ndarray
                mAP_ground_truth_in = true_boxes.numpy().reshape(-1, 4)[:, ::-1]

                mAP_predict_in = mAP_predict_in.detach().cpu().numpy()
                mAP_predict_in[:, [1, 3]] = mAP_predict_in[:, [1, 3]] - dh
                mAP_predict_in[:, [0, 2]] = mAP_predict_in[:, [0, 2]] - dw
                mAP_predict_in = mAP_predict_in / ratio

                mAP_predicts.append(mAP_predict_in)
                mAP_ground_truths.append(mAP_ground_truth_in)

        MAP = mAP(mAP_predicts, mAP_ground_truths, 0.5)
        self.logger.info('AP: %.2f %%' % (MAP.elevenPointAP * 100))

    def add_train_summary(self, loss_dict, step):
        self.writer.add_scalar(tag=f'Train/total_loss', scalar_value=loss_dict['total_loss'], global_step=step)
        self.writer.add_scalar(f'Train/xy_loss', loss_dict['xy_loss'], step)
        self.writer.add_scalar(f'Train/wh_loss', loss_dict['wh_loss'], step)
        self.writer.add_scalar(f'Train/class_loss', loss_dict['class_loss'], step)
        self.writer.add_scalar(f'Train/obj_conf_loss', loss_dict['obj_conf_loss'], step)
        self.writer.add_scalar(f'Train/noobj_conf_loss', loss_dict['noobj_conf_loss'], step)

    def add_test_summary(self, pred_dict, step):
        self.writer.add_histogram('Test/score', pred_dict['score'], global_step=step)
        self.writer.add_histogram('Test/conf', pred_dict['conf'], step)
        self.writer.add_histogram('Test/cls', pred_dict['cls'], step)


if __name__ == '__main__':
    model = Yolov3()
    model.test()



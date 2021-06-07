# ---------------------------------------------------------------------------- #
# Pytorch GPA Cross-domain Detection
# Witten by Minghao Xu, Hang Wang
# Based on the Faster R-CNN code written by Jianwei Yang
# ---------------------------------------------------------------------------- #
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pprint
import pdb
import time
import math
import _init_paths

import torch
import torch.nn as nn
from torch.autograd import Variable

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader

from model.utils.parser_func import parse_args, set_dataset_args
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import sampler, save_checkpoint, adjust_learning_rate, clip_gradient, get_lr_at_iter


# Cosine annealing learning rate
def cosine_da_weight(base_weight, curr_epoch, max_epoch):
    return base_weight * (1 + math.cos(math.pi * min(curr_epoch - 1, max_epoch) / max_epoch)) / 2


if __name__ == '__main__':

    args = parse_args()
    print('Called with args:')
    print(args)
    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    ########
    cfg.TRAIN.RPN_FG_FRACTION = args.pos_ratio
    cfg.TRAIN.RPN_BATCHSIZE = args.rpn_bs
    cfg.TRAIN.BATCH_SIZE = args.train_bs
    print('RPN_FG_FRACTION:', cfg.TRAIN.RPN_FG_FRACTION)
    print('RPN_BATCHSIZE:', cfg.TRAIN.RPN_BATCHSIZE)
    print('BATCH_SIZE', cfg.TRAIN.BATCH_SIZE)
    ########
    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)
    ########
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed(cfg.RNG_SEED)
    ########
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    cfg.TRAIN.USE_FLIPPED = False
    cfg.USE_GPU_NMS = args.cuda

    # for source domain
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    # for target domain
    tgt_imdb, tgt_roidb, tgt_ratio_list, tgt_ratio_index = combined_roidb(args.imdb_name_target)
    tgt_train_size = len(tgt_roidb)
    print('{:d} roidb entries for source domain'.format(len(roidb)))
    print('{:d} roidb entries for target domain'.format(len(tgt_roidb)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # define the dataloader
    sampler_batch = sampler(train_size, args.batch_size)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, imdb.num_classes, training=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler_batch, num_workers=args.num_workers)

    tgt_sampler_batch = sampler(tgt_train_size, args.batch_size)
    tgt_dataset = roibatchLoader(tgt_roidb, tgt_ratio_list, tgt_ratio_index, args.batch_size, tgt_imdb.num_classes, training=True)
    tgt_dataloader = torch.utils.data.DataLoader(tgt_dataset, batch_size=args.batch_size, sampler=tgt_sampler_batch, num_workers=args.num_workers)

    # initialize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    tgt_im_data = torch.FloatTensor(1)
    tgt_im_info = torch.FloatTensor(1)
    tgt_num_boxes = torch.FloatTensor(1)
    tgt_gt_boxes = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        tgt_im_data = tgt_im_data.cuda()
        tgt_im_info = tgt_im_info.cuda()
        tgt_num_boxes = tgt_num_boxes.cuda()
        tgt_gt_boxes = tgt_gt_boxes.cuda()
    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    tgt_im_data = Variable(tgt_im_data)
    tgt_im_info = Variable(tgt_im_info)
    tgt_num_boxes = Variable(tgt_num_boxes)
    tgt_gt_boxes = Variable(tgt_gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    # initialize the network here.
    from model.faster_rcnn.vgg16_GPA import vgg16
    from model.faster_rcnn.resnet_GPA import resnet

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, mode=args.mode, rpn_mode=args.rpn_mode)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic, mode=args.mode, rpn_mode=args.rpn_mode)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic, mode=args.mode, rpn_mode=args.rpn_mode)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()

    if args.resume:
        print("loading checkpoint %s" % args.load_name)
        checkpoint = torch.load(args.load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % args.load_name)

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    # iters_per_epoch = int(train_size / args.batch_size)
    iters_per_epoch = int(10000 / args.batch_size)
    # tgt_iters_per_epoch = int(tgt_train_size / args.batch_size)
    tgt_iters_per_epoch = int(10000 / args.batch_size)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter

        logger = SummaryWriter("logs")

    lr_decay_step = sorted([int(decay_step) for decay_step in args.lr_decay_steps.split(',') if decay_step.strip()])  # lr

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        while lr_decay_step and epoch > lr_decay_step[0]:
            lr_decay_step.pop(0)
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma
        base_lr = lr

        data_iter = iter(dataloader)
        tgt_data_iter = iter(tgt_dataloader)
        for step in range(iters_per_epoch):
            if epoch == 1 and step <= args.warm_up:
                lr = base_lr * get_lr_at_iter(step / args.warm_up)
            else:
                lr = base_lr

            data = next(data_iter)
            # put source data into variable
            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])

            if step % tgt_iters_per_epoch == 0:
                tgt_data_iter = iter(tgt_dataloader)
            tgt_data = next(tgt_data_iter)
            # put target data into variable
            tgt_im_data.resize_(tgt_data[0].size()).copy_(tgt_data[0])
            tgt_im_info.resize_(tgt_data[1].size()).copy_(tgt_data[1])
            tgt_gt_boxes.resize_(tgt_data[2].size()).copy_(tgt_data[2])
            tgt_num_boxes.resize_(tgt_data[3].size()).copy_(tgt_data[3])

            fasterRCNN.zero_grad()

            rois, tgt_rois, cls_prob, tgt_cls_prob, bbox_pred, tgt_bbox_pred, \
            rpn_loss_cls, _, rpn_loss_box, _, \
            RCNN_loss_cls, _, RCNN_loss_bbox, _, \
            RCNN_loss_intra, RCNN_loss_inter, \
            rois_label, tgt_rois_label, \
            RPN_loss_intra, RPN_loss_inter = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes)

            # adjust RPN's domain adaptation weight / fix it as constant
            if args.cosine_rpn_da_weight:
                rpn_da_weight = cosine_da_weight(args.rpn_da_weight, epoch, args.max_epochs)
            else:
                rpn_da_weight = args.rpn_da_weight

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
                   + args.da_weight * (RCNN_loss_intra + RCNN_loss_inter) \
                   + rpn_da_weight * (RPN_loss_intra + RPN_loss_inter)

            if args.mGPUs:
                loss_temp = loss.mean().item()
            else:
                loss_temp = loss.item()

            # backward
            optimizer.zero_grad()
            if args.mGPUs:
                loss = loss.mean()
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 10.)
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()

                    intra_loss = RCNN_loss_intra.mean().item()
                    inter_loss = RCNN_loss_inter.mean().item()

                    rpn_intra_loss = RPN_loss_intra.mean().item()
                    rpn_inter_loss = RPN_loss_inter.mean().item()

                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                    tgt_fg_cnt = torch.sum(tgt_rois_label.data.ne(0))
                    tgt_bg_cnt = tgt_rois_label.data.numel() - tgt_fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()

                    intra_loss = RCNN_loss_intra.item()
                    inter_loss = RCNN_loss_inter.item()

                    rpn_intra_loss = RPN_loss_intra.item()
                    rpn_inter_loss = RPN_loss_inter.item()

                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                    tgt_fg_cnt = torch.sum(tgt_rois_label.data.ne(0))
                    tgt_bg_cnt = tgt_rois_label.data.numel() - tgt_fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), tgt_fg/tgt_bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, tgt_fg_cnt, tgt_bg_cnt, end - start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                print("\t\t\tintra_loss: %.4f, inter_loss: %.4f" % (intra_loss, inter_loss))
                print("\t\t\trpn_intra_loss: %.4f, rpn_inter_loss: %.4f" % (rpn_intra_loss, rpn_inter_loss))

                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)
                loss_temp = 0
                start = time.time()

        save_name = os.path.join(output_dir,
                                 'GPA_{}_mode_{}_{}_rpn_mode_{}_{}_session_{}_epoch_{}_step_{}.pth'.format(
                                     args.dataset_t, args.mode, args.da_weight, args.rpn_mode, args.rpn_da_weight, args.session, epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger.close()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn_GPA import _fasterRCNN


class vgg16(_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False,
                 mode='adapt', rpn_mode='adapt'):  ####

        self.model_path = cfg.VGG16_PATH
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        self.mode = mode  ####
        self.rpn_mode = rpn_mode  ####

        _fasterRCNN.__init__(self, classes, class_agnostic,
                             mode, rpn_mode)  ####

    def _init_modules(self):
        vgg = models.vgg16()

        if self.pretrained:
            print("Loading pretrained weights from %s" % self.model_path)
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        self.rpn_adapt_feat = nn.Linear(cfg.POOLING_SIZE * cfg.POOLING_SIZE * 512, 128)  ####
        self.RCNN_adapt_feat = nn.Linear(4096, 64)  ####

        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

        self.RCNN_top = vgg.classifier

        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

    def _head_to_tail(self, pool5):
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)
        return fc7

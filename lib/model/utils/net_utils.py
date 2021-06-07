# coding:utf-8

import numpy as np
import cv2
import pdb
import random
import math
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable, Function
from torch.utils.data.sampler import Sampler
import torchvision.models as models

from model.utils.config import cfg
from model.roi_crop.functions.roi_crop import RoICropFunction


class sampler(Sampler):
    def __init__(self, train_size, batch_size):

        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range
        self.rand_num_view = self.rand_num.view(-1)
        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)
        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


class EFocalLoss(nn.Module):
    """
    This criterion is a implemenation of Focal Loss, which is proposed in
    "Focal Loss for Dense Object Detection".
           Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

    The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(EFocalLoss, self).__init__()

        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        # inputs = F.sigmoid(inputs)
        P = F.softmax(inputs)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)
        batch_loss = -alpha * torch.exp(-self.gamma * probs) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalLoss(nn.Module):
    """
    This criterion is a implemenation of Focal Loss, which is proposed in
    "Focal Loss for Dense Object Detection".
           Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, sigmoid=False, reduce=True):
        super(FocalLoss, self).__init__()

        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.sigmoid = sigmoid
        self.reduce = reduce

    def forward(self, inputs, targets):
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        if self.sigmoid:
            P = F.sigmoid(inputs)
            # F.softmax(inputs)
            if targets == 0:
                probs = 1 - P  # (P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
            if targets == 1:
                probs = P  # (P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
        else:
            # inputs = F.sigmoid(inputs)
            P = F.softmax(inputs)

            class_mask = inputs.data.new(N, C).fill_(0)
            class_mask = Variable(class_mask)
            ids = targets.view(-1, 1)
            class_mask.scatter_(1, ids.data, 1.)
            # print(class_mask)

            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.cuda()
            alpha = self.alpha[ids.data.view(-1)]

            probs = (P * class_mask).sum(1).view(-1, 1)

            log_p = probs.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)

            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
            # print('-----bacth_loss------')
            # print(batch_loss)

        if not self.reduce:
            return batch_loss
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalPseudo(nn.Module):
    """
    This criterion is a implemenation of Focal Loss, which is proposed in
    "Focal Loss for Dense Object Detection".
           Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

    The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, threshold=0.8):
        super(FocalPseudo, self).__init__()

        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.threshold = threshold

    def forward(self, inputs):
        N = inputs.size(0)
        C = inputs.size(1)
        inputs = inputs[0, :, :]
        # print(inputs)
        # pdb.set_trace()
        inputs, ind = torch.max(inputs, 1)
        ones = torch.ones(inputs.size()).cuda()
        value = torch.where(inputs > self.threshold, inputs, ones)
        # pdb.set_trace()
        # print(value)
        try:
            ind = value.ne(1)
            indexes = torch.nonzero(ind)
            # value2 = inputs[indexes]
            inputs = inputs[indexes]
            log_p = inputs.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)
            if not self.gamma == 0:
                batch_loss = - (torch.pow((1 - inputs), self.gamma)) * log_p
            else:
                batch_loss = - log_p
        except:
            # inputs = inputs#[indexes]
            log_p = value.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)
            if not self.gamma == 0:
                batch_loss = - (torch.pow((1 - inputs), self.gamma)) * log_p
            else:
                batch_loss = - log_p
        # print('-----bacth_loss------')
        # print(batch_loss)
        # batch_loss = batch_loss #* weight
        if self.size_average:
            try:
                loss = batch_loss.mean()  # + 0.1*balance
            except:
                pdb.set_trace()
        else:
            loss = batch_loss.sum()
        return loss


def CrossEntropy(output, label):
    criteria = torch.nn.CrossEntropyLoss()
    loss = criteria(output, label)
    return loss


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        # pdb.set_trace()
        return grad_output * -self.lambd


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


# gcn.layers
################################################################################
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features).cuda())

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features).cuda())
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_features) + '->' + str(self.out_features) + ')'


# gcn.models
################################################################################
class GCN(nn.Module):
    def __init__(self, nproposal, nfeat, dropout=0.5):
        super(GCN, self).__init__()

        self.dropout = dropout
        self.conv_1 = GraphConvolution(nfeat, nfeat)
        self.emb = GraphConvolution(nfeat, nfeat)
        # self.weight = GraphConvolution(nfeat, 1)
        self.weight = GraphConvolution(1, 1)

    def forward(self, x, w, adj):
        feat_1 = F.relu(self.conv_1(x, adj))
        feat_1 = F.dropout(feat_1, self.dropout, training=self.training)
        emb = self.emb(feat_1, adj)
        # weight = F.sigmoid(self.weight(feat_1, adj))
        # weight = torch.mm(adj, w)
        weight = w

        output = emb * weight
        # output = torch.sum(output, dim = 0) / torch.sum(weight)

        return output, weight


class GCN_pooling(nn.Module):
    def __init__(self, nproposal, nfeat, dropout=0.5):
        super(GCN, self).__init__()

        self.dropout = dropout

        # the first group
        nproposal_1 = nproposal / 4
        self.emb_1 = GraphConvolution(nfeat, nfeat)
        self.pool_1 = GraphConvolution(nfeat, nproposal_1)

        # the second group
        nproposal_2 = nproposal / 16
        self.emb_2 = GraphConvolution(nfeat, nfeat)
        self.pool_2 = GraphConvolution(nfeat, nproposal_2)

        # the third group
        nproposal_3 = 1
        self.emb_3 = GraphConvolution(nfeat, nfeat)
        self.pool_3 = GraphConvolution(nfeat, nproposal_3)

    def forward(self, x, adj):
        # the first group
        input_1 = F.relu(self.emb_1(x, adj))
        # input_1 = F.dropout(input_1, self.dropout, training = self.training)
        assign_1 = F.softmax(F.relu(self.pool_1(x, adj)), dim=1)
        # assign_1 = F.dropout(assign_1, self.dropout, training = self.training)
        output_1 = torch.mm(assign_1.t(), input_1)
        adj_1 = torch.mm(torch.mm(assign_1.t(), adj), assign_1)

        # the second group
        input_2 = F.relu(self.emb_2(output_1, adj_1))
        # input_2 = F.dropout(input_2, self.dropout, training = self.training)
        assign_2 = F.softmax(F.relu(self.pool_2(output_1, adj_1)), dim=1)
        # assign_2 = F.dropout(assign_2, self.dropout, training = self.training)
        output_2 = torch.mm(assign_2.t(), input_2)
        adj_2 = torch.mm(torch.mm(assign_2.t(), adj_1), assign_2)

        # the third group 
        input_3 = self.emb_3(output_2, adj_2)
        assign_3 = F.softmax(self.pool_3(output_2, adj_2), dim=1)
        output_3 = torch.mm(assign_3.t(), input_3)

        # get the link and entropy regularization loss
        gcn_loss_link = 0
        gcn_loss_entropy = 0

        gcn_loss_link = gcn_loss_link + torch.pow(adj - torch.mm(assign_1, assign_1.t()), 2).mean()
        gcn_loss_link = gcn_loss_link + torch.pow(adj_1 - torch.mm(assign_2, assign_2.t()), 2).mean()
        gcn_loss_link = gcn_loss_link + torch.pow(adj_2 - torch.mm(assign_3, assign_3.t()), 2).mean()

        entropy_1 = -torch.mul(assign_1, torch.log(assign_1))
        gcn_loss_entropy = gcn_loss_entropy + torch.mean(torch.sum(entropy_1, dim=1), dim=0)
        entropy_2 = -torch.mul(assign_2, torch.log(assign_2))
        gcn_loss_entropy = gcn_loss_entropy + torch.mean(torch.sum(entropy_2, dim=1), dim=0)
        entropy_3 = -torch.mul(assign_3, torch.log(assign_3))
        gcn_loss_entropy = gcn_loss_entropy + torch.mean(torch.sum(entropy_3, dim=1), dim=0)

        return output_3.view(-1), gcn_loss_link, gcn_loss_entropy


# gcn.utils
################################################################################
def get_adj(rois, epsilon=1e-6):
    # compute the area of every bbox
    area = (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2])
    area = area + (area == 0).float() * epsilon

    # compute iou
    x_min = rois[:, 1]
    x_min_copy = torch.stack([x_min] * rois.size(0), dim=0)
    x_min_copy_ = x_min_copy.permute((1, 0))
    x_min_matrix = torch.max(torch.stack([x_min_copy, x_min_copy_], dim=-1), dim=-1)[0]

    x_max = rois[:, 3]
    x_max_copy = torch.stack([x_max] * rois.size(0), dim=0)
    x_max_copy_ = x_max_copy.permute((1, 0))
    x_max_matrix = torch.min(torch.stack([x_max_copy, x_max_copy_], dim=-1), dim=-1)[0]

    y_min = rois[:, 2]
    y_min_copy = torch.stack([y_min] * rois.size(0), dim=0)
    y_min_copy_ = y_min_copy.permute((1, 0))
    y_min_matrix = torch.max(torch.stack([y_min_copy, y_min_copy_], dim=-1), dim=-1)[0]

    y_max = rois[:, 4]
    y_max_copy = torch.stack([y_max] * rois.size(0), dim=0)
    y_max_copy_ = y_max_copy.permute((1, 0))
    y_max_matrix = torch.min(torch.stack([y_max_copy, y_max_copy_], dim=-1), dim=-1)[0]

    w = torch.max(torch.stack([(x_max_matrix - x_min_matrix), torch.zeros_like(x_min_matrix)], dim=-1), dim=-1)[0]
    h = torch.max(torch.stack([(y_max_matrix - y_min_matrix), torch.zeros_like(y_min_matrix)], dim=-1), dim=-1)[0]
    intersection = w * h

    area_copy = torch.stack([area] * rois.size(0), dim=0)
    area_copy_ = area_copy.permute((1, 0))
    area_sum = area_copy + area_copy_

    union = area_sum - intersection
    iou = intersection / union

    return iou


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """ Load citation network dataset (cora only for now) """
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """ Row-normalize sparse matrix """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """ Convert a scipy sparse matrix to a torch sparse tensor. """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    np_indices = np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    indices = torch.from_numpy(np_indices)
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_lr_at_iter(alpha):
    return 1. / 3. * (1 - alpha) + alpha


################################################################################


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """ Computes a gradient clipping coefficient based on gradient norm. """
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm).item()

    norm = (clip_norm / max(totalnorm, clip_norm))
    # print(norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)


def vis_detections(im, class_name, dets, thresh=0.8):
    """ Visual debugging of detections. """
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 0, 255), 1)
            # cv2.rectangle(im, (bbox[0], bbox[1] - 45), (bbox[0]+250, bbox[1] + 5), (255, 0, 0), thickness=-1)
            cv2.putText(im, '%s: %.2f' % (class_name, score), (bbox[0], bbox[1] - 6), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=1)
        # if score > thresh:
        #    cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
        #    cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)
    return im


def adjust_learning_rate(optimizer, decay=0.1):
    """ Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs """
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


# def adjust_learning_rate(optimizer, decay=0.1, lr_init = 0.001):
#    """ Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs """
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = decay * lr_init


import math


def calc_supp(iter, iter_total=80000):
    p = float(iter) / iter_total
    # print(math.exp(-10*p))
    return 2 / (1 + math.exp(-10 * p)) - 1


def save_checkpoint(state, filename):
    torch.save(state, filename)


def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box


def _crop_pool_layer(bottom, rois, max_pool=True):
    # code modified from 
    # https://github.com/ruotianluo/pytorch-faster-rcnn
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    rois = rois.detach()
    batch_size = bottom.size(0)
    D = bottom.size(1)
    H = bottom.size(2)
    W = bottom.size(3)
    roi_per_batch = rois.size(0) / batch_size
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = bottom.size(2)
    width = bottom.size(3)

    # affine theta
    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([
        (x2 - x1) / (width - 1),
        zero,
        (x1 + x2 - width + 1) / (width - 1),
        zero,
        (y2 - y1) / (height - 1),
        (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    if max_pool:
        pre_pool_size = cfg.POOLING_SIZE * 2
        grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)))
        bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W) \
            .contiguous().view(-1, D, H, W)
        crops = F.grid_sample(bottom, grid)
        crops = F.max_pool2d(crops, 2, 2)
    else:
        grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, cfg.POOLING_SIZE, cfg.POOLING_SIZE)))
        bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W) \
            .contiguous().view(-1, D, H, W)
        crops = F.grid_sample(bottom, grid)

    return crops, grid


def _affine_grid_gen(rois, input_size, grid_size):
    rois = rois.detach()

    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([
        (x2 - x1) / (width - 1),
        zero,
        (x1 + x2 - width + 1) / (width - 1),
        zero,
        (y2 - y1) / (height - 1),
        (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))

    return grid


def _affine_theta(rois, input_size):
    rois = rois.detach()

    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())

    # theta = torch.cat([
    #  (x2 - x1) / (width - 1),
    #  zero,
    #  (x1 + x2 - width + 1) / (width - 1),
    #  zero,
    #  (y2 - y1) / (height - 1),
    #  (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    theta = torch.cat([
        (y2 - y1) / (height - 1),
        zero,
        (y1 + y2 - height + 1) / (height - 1),
        zero,
        (x2 - x1) / (width - 1),
        (x1 + x2 - width + 1) / (width - 1)], 1).view(-1, 2, 3)

    return theta


def compare_grid_sample():
    # do gradcheck
    N = random.randint(1, 8)
    C = 2  # random.randint(1, 8)
    H = 5  # random.randint(1, 8)
    W = 4  # random.randint(1, 8)
    input = Variable(torch.randn(N, C, H, W).cuda(), requires_grad=True)
    input_p = input.clone().data.contiguous()

    grid = Variable(torch.randn(N, H, W, 2).cuda(), requires_grad=True)
    grid_clone = grid.clone().contiguous()

    out_offcial = F.grid_sample(input, grid)
    grad_outputs = Variable(torch.rand(out_offcial.size()).cuda())
    grad_outputs_clone = grad_outputs.clone().contiguous()
    grad_inputs = torch.autograd.grad(out_offcial, (input, grid), grad_outputs.contiguous())
    grad_input_off = grad_inputs[0]

    crf = RoICropFunction()
    grid_yx = torch.stack([grid_clone.data[:, :, :, 1], grid_clone.data[:, :, :, 0]], 3).contiguous().cuda()
    out_stn = crf.forward(input_p, grid_yx)
    grad_inputs = crf.backward(grad_outputs_clone.data)
    grad_input_stn = grad_inputs[0]
    pdb.set_trace()

    delta = (grad_input_off.data - grad_input_stn).sum()

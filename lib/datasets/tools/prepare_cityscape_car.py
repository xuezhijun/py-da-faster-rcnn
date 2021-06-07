# -* coding: utf-8 -*-

import os
import cv2
import numpy as np
from tqdm import tqdm

from writexml import writexml

# specify path
data_dir = 'CITYSCAPES_DIR'  # cityscapes dataset path

# initialization
img_dir = 'VOC2007/JPEGImages'
sets_dir = 'VOC2007/ImageSets/Main'
annotation_dir = 'VOC2007/Annotations'

if not os.path.exists(img_dir):
    os.makedirs(img_dir)
if not os.path.exists(sets_dir):
    os.makedirs(sets_dir)
if not os.path.exists(annotation_dir):
    os.makedirs(annotation_dir)

# organize images & prepare split list.
# process train images
source_img_path = os.path.join(data_dir, 'leftImg8bit', 'train')
train_list = []
print('process source train images')

for root, dirs, files in os.walk(source_img_path, topdown=True):
    files = [name for name in files if name.endswith('.png')]
    for name in tqdm(files):
        im_name = name.replace('.png', '')
        img = cv2.imread(os.path.join(root, name))
        cv2.imwrite(os.path.join(img_dir, im_name + '.jpg'), img)
        train_list.append(im_name)

# process val images
print('process target test images')
target_img_path = os.path.join(data_dir, 'leftImg8bit', 'val')
val_list = []

for root, dirs, files in os.walk(target_img_path, topdown=True):
    files = [name for name in files if name.endswith('.png')]
    for name in tqdm(files):
        im_name = name.replace('.png', '')
        img = cv2.imread(os.path.join(root, name))
        cv2.imwrite(os.path.join(img_dir, im_name + '.jpg'), img)
        val_list.append(im_name)

# write the list
with open(os.path.join(sets_dir, "train.txt"), 'w') as f:
    for item in train_list:
        f.write("{}\n".format(item))

with open(os.path.join(sets_dir, "val.txt"), 'w') as f:
    for item in val_list:
        f.write("{}\n".format(item))

# prepare the annotation needed for training/testing.
cityscapes_semantics = ['ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground', 'road',
                        'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge',
                        'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                        'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle',
                        'bicycle']
instance_semantics = ['car']

bind = {}
for i, elt in enumerate(instance_semantics):
    if elt not in bind:
        bind[elt] = i
lb_filter = [bind.get(itm, -1) for itm in cityscapes_semantics]

# instanceIds.png
source_img_path = os.path.join(data_dir, 'gtFine')
for root, dirs, files in os.walk(source_img_path, topdown=True):
    files = [name for name in files if name.endswith('instanceIds.png')]
    for name in tqdm(files):
        im_name = name.replace('_gtFine_instanceIds.png', '_leftImg8bit')
        im_inst = cv2.imread(os.path.join(root, name), cv2.IMREAD_ANYDEPTH)
        im_lb = cv2.imread(os.path.join(root, name.replace('_gtFine_instanceIds.png', '_gtFine_labelIds.png')), cv2.IMREAD_ANYDEPTH)

        all_inst_id = np.setdiff1d(np.unique(im_inst), 0)
        boxes = []
        categories = []
        for i_inst in all_inst_id:
            inst_mask = (im_inst == i_inst)
            inst_mask_int = inst_mask - 0
            # print(np.unique(inst_mask))
            assert (len(np.unique(inst_mask_int)) == 2)
            x_cods = np.where(np.sum(inst_mask_int, 0) > 0)
            y_cods = np.where(np.sum(inst_mask_int, 1) > 0)
            box = [np.min(x_cods), np.min(y_cods), np.max(x_cods), np.max(y_cods)]
            boxes.append(box)
            # print(box)
            inst_lb = np.unique(im_lb[inst_mask])
            # print(inst_lb)
            category = lb_filter[inst_lb[0] - 1]
            categories.append(category + 1)
            # break
        # plus 1, in order to match matlab code
        boxes = np.array(boxes) + 1
        categories = np.array(categories)
        boxes = boxes[categories != 0]
        categories = categories[categories != 0]
        filename = im_name + '.jpg'
        img_shape = (*im_inst.shape, 3)
        writexml(filename, img_shape, boxes, categories, instance_semantics,
                 os.path.join(annotation_dir, "{}.xml".format(im_name)))
# you need move generated 'VOC2007' directory into data/cityscape_car

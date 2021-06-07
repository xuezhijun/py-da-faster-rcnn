# ---------------------------------------------------------------------------- #
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# ---------------------------------------------------------------------------- #
""" Factory method for easily getting imdbs by name. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_water import pascal_voc_water
from datasets.pascal_voc_cyclewater import pascal_voc_cyclewater
from datasets.pascal_voc_cycleclipart import pascal_voc_cycleclipart

from datasets.water import water
from datasets.clipart import clipart

from datasets.sim10k import sim10k
from datasets.sim10k_cycle import sim10k_cycle

from datasets.kitti_car import kitti_car

from datasets.cityscape import cityscape
from datasets.cityscape_car import cityscape_car
from datasets.foggy_cityscape import foggy_cityscape

from datasets.init_sunny import init_sunny
from datasets.init_night import init_night
from datasets.init_rainy import init_rainy
from datasets.init_cloudy import init_cloudy

from datasets.synthia import synthia
from datasets.cityscape_synthia import cityscape_synthia

# from datasets.syn2real import syn2real
# from datasets.coco_syn2real import coco_syn2real

# from datasets.sim10k_coco import sim10k_coco
# from datasets.cityscape_car_coco import cityscape_car_coco

# from datasets.imagenet import imagenet
# from datasets.vg import vg


# Set up pascal_voc
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_water_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc_water(split, year))

for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_cyclewater_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc_cyclewater(split, year))
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_cycleclipart_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc_cycleclipart(split, year))

# Set up watercolor voc style
for year in ['2007']:
    for split in ['train', 'test']:
        name = 'water_{}'.format(split)
        __sets[name] = (lambda split=split: water(split, year))
# Set up clipart voc style
for year in ['2007']:
    for split in ['trainval', 'test']:
        name = 'clipart_{}'.format(split)
        __sets[name] = (lambda split=split: clipart(split, year))

# Set up sim10k voc style
for split in ['train', 'val']:
    name = 'sim10k_{}'.format(split)
    __sets[name] = (lambda split=split: sim10k(split))
for split in ['train']:
    name = 'sim10k_cycle_{}'.format(split)
    __sets[name] = (lambda split=split: sim10k_cycle(split))

# Set up cityscape voc style
for split in ['train', 'val', 'test']:
    name = 'cityscape_{}'.format(split)
    __sets[name] = (lambda split=split: cityscape(split))
# Set up cityscape_car voc style
for split in ['train', 'val', 'test']:
    name = 'cityscape_car_{}'.format(split)
    __sets[name] = (lambda split=split: cityscape_car(split))
# Set up foggy cityscape voc style
for split in ['train', 'val', 'test']:
    name = 'foggy_cityscape_{}'.format(split)
    __sets[name] = (lambda split=split: foggy_cityscape(split))

# Set up KITTI voc style
for split in ['train']:
    name = 'kitti_car_{}'.format(split)
    __sets[name] = (lambda split=split: kitti_car(split))

# Set up INIT voc style
for split in ['trainval']:
    name = 'init_sunny_{}'.format(split)
    __sets[name] = (lambda split=split: init_sunny(split))
for split in ['train', 'val']:
    name = 'init_night_{}'.format(split)
    __sets[name] = (lambda split=split: init_night(split))
for split in ['train', 'val']:
    name = 'init_rainy_{}'.format(split)
    __sets[name] = (lambda split=split: init_rainy(split))
for split in ['train', 'val']:
    name = 'init_cloudy_{}'.format(split)
    __sets[name] = (lambda split=split: init_cloudy(split))

# Set up SYNTHIA-RAND-CITYSCAPES voc style
for split in ['train']:
    name = 'synthia_{}'.format(split)
    __sets[name] = (lambda split=split: synthia(split))
for split in ['train', 'val']:
    name = 'cityscape_synthia_{}'.format(split)
    __sets[name] = (lambda split=split: cityscape_synthia(split))

'''
# Set up coco_2014_<split>
for year in ["2014"]:
    for split in ["train", "val", "minival", "valminusminival", "trainval"]:
        name = "coco_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: coco(split, year)

# Set up coco_2014_cap_<split>
for year in ["2014"]:
    for split in ["train", "val", "capval", "valminuscapval", "trainval"]:
        name = "coco_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: coco(split, year)

# Set up coco_2015_<split>
for year in ["2015"]:
    for split in ["test", "test-dev"]:
        name = "coco_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: coco(split, year)

# Set up sim10k coco style
for year in ["2019"]:
    for split in ["train", "val"]:
        name = "sim10k_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: sim10k_coco(split, year)

# Set up cityscape_car coco style
for year in ["2019"]:
    for split in ["train", "val"]:
        name = "cityscapes_car_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: cityscape_car_coco(split, year)

# Set up vg_<split>
for version in ['1600-400-20']:
    for split in ['minitrain', 'train', 'minival', 'val', 'test']:
        name = 'vg_{}_{}'.format(version,split)
        __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in ["150-50-20", "150-50-50", "500-150-80", "750-250-150", "1750-700-450", "1600-400-20"]:
    for split in ["minitrain", "smalltrain", "train", "minival", "smallval", "val", "test"]:
        name = "vg_{}_{}".format(version, split)
        __sets[name] = lambda split=split, version=version: vg(version, split)

# set up imagenet.
for split in ["train", "val", "val1", "val2", "test"]:
    name = "imagenet_{}".format(split)
    devkit_path = "data/imagenet/ILSVRC/devkit"
    data_path = "data/imagenet/ILSVRC"
    __sets[name] = lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split, devkit_path, data_path)
'''


def get_imdb(name):
    """ Get an imdb (image database) by name. """
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """ List all registered imdbs. """
    return list(__sets.keys())

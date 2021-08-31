# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.icub import icub_dataset
import numpy as np

#icub transformation dataset
icub_manual_path = '/home/IIT.LOCAL/emaiettini/workspace/Repos/SSM/data/iCubWorld-Transformations_manual'
# icub_devkit_path = '/home/elisa/Repos/py-faster_icubworld/data/iCubWorld-Translation_devkit'

for t in ['train', 'test']:
    for task in ['TASK2']:
        for numObjs in ['30', '21']:
            for supervision in ['supervised', 'unsupervised']:
                split = '{}_{}_{}objs_{}'.format(t, task, numObjs, supervision)
                name = '{}_{}_{}objs_{}'.format(t, task, numObjs, supervision)
                __sets[name] = (lambda split=split: icub_dataset(split))

for t in ['train']:
    for task in ['TASK2', 'TASK2_EAL']:
        for numObjs in ['21']:
            for color in ['white', 'pois_odd', 'green', 'full']:
                split = '{}_{}_{}objs_{}'.format(t, task, numObjs, color)
                name = '{}_{}_{}objs_{}'.format(t, task, numObjs, color)
                __sets[name] = (lambda split=split: icub_dataset(split))

for t in ['test']:
    for task in ['TASK2', 'TASK2_EAL']:
        for numObjs in ['21']:
            for color in ['white', 'pois_odd', 'green']:
                split = '{}_{}_{}objs_{}'.format(t, task, numObjs, color)
                name = '{}_{}_{}objs_{}'.format(t, task, numObjs, color)
                __sets[name] = (lambda split=split: icub_dataset(split, devkit_path=icub_manual_path))

for t in ['train', 'test']:
    for task in ['TASK2']:
        for numObjs in ['21']:
                split = '{}_{}_{}objs'.format(t, task, numObjs, supervision)
                name = '{}_{}_{}objs'.format(t, task, numObjs, supervision)
                __sets[name] = (lambda split=split: icub_dataset(split))


for t in ['train', 'test']:
    for task in ['TASK2']:
        for numObjs in ['21']:
                for supervision in ['acquisition1', 'acquisition2']:
                    split = '{}_{}_{}objs_{}'.format(t, task, numObjs, supervision)
                    name = '{}_{}_{}objs_{}'.format(t, task, numObjs, supervision)
                    __sets[name] = (lambda split=split: icub_dataset(split, devkit_path=icub_manual_path))

for t in ['train']:
    for task in ['TASK2', 'TASK2_sequence']:
        for numObjs in ['30', '21', '5', '10']:
                for supervision in ['TRA_supervised', 'NoTRA_unsupervised', 'NoTRA_unsupervised_ordered']:
                    split = '{}_{}_{}objs_{}'.format(t, task, numObjs, supervision)
                    name = '{}_{}_{}objs_{}'.format(t, task, numObjs, supervision)
                    __sets[name] = (lambda split=split: icub_dataset(split))

for t in ['train']:
    for task in ['TASK2']:
        for numObjs in ['30']:
                for supervision in ['supervised', 'unsupervised']:
                    split = '{}_{}_{}objs_1over8_{}_fake'.format(t, task, numObjs, supervision)
                    name = '{}_{}_{}objs_1over8_{}_fake'.format(t, task, numObjs, supervision)
                    __sets[name] = (lambda split=split: icub_dataset(split))
  
for t in ['test']: # split sarebbe l'imageset
    for task in ['TASK2']:
        for numObjs in ['30', '5', '10']:
            split = '{}_{}_{}objs_manual'.format(t, task, numObjs)
            name = '{}_{}_{}objs_manual'.format(t, task, numObjs)
            __sets[name] = (lambda split=split: icub_dataset(split, devkit_path=icub_manual_path))

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012', '0712']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()

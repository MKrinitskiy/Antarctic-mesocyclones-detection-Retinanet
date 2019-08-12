"""
Copyright 2017-2018 yhenon (https://github.com/yhenon/)
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from .generator import Generator
from ..utils.image import read_image_bgr

import numpy as np
from PIL import Image
from six import raise_from

import csv
import sys
import os.path
import pickle
from datetime import datetime

from keras_retinanet.mk_helpers import find_files


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_reader):
    """ Parse the classes file given by csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


class SAIL_Generator(Generator):
    """ Generate data for a custom CSV dataset.

    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """

    def __init__(self,
                 sail_annotations_file,
                 csv_class_file,
                 dataset_type = 'train',
                 **kwargs):
        """ Initialize a CSV data generator.

        Args
            csv_data_file: Path to the CSV annotations file.
            csv_class_file: Path to the CSV classes file.
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
        """

        self.annotations_file = os.path.abspath(sail_annotations_file)
        self.class_file = os.path.abspath(csv_class_file)
        self.dataset_type = dataset_type

        total_examples_number = None
        if (('val_steps' in kwargs.keys()) &  ('val_batch_size' in kwargs.keys())):
            if ((kwargs['val_steps'] is not None) & (kwargs['val_batch_size'] is not None)):
                total_examples_number = kwargs['val_steps']*kwargs['val_batch_size']

        self.data_manager = data_manager(self.annotations_file, self.dataset_type, total_examples_number)
        self.image_names = []

        self.__len__value = False
        if 'val_steps' in kwargs.keys():
            if kwargs['val_steps'] is not None:
                self.val_steps = kwargs['val_steps']
                # self.__len__value = True
                del kwargs['val_steps']

        if 'force_steps_per_epoch' in kwargs.keys():
            if kwargs['force_steps_per_epoch'] is not None:
                self.force_steps_per_epoch = kwargs['force_steps_per_epoch']
                self.__len__value = True
                del kwargs['force_steps_per_epoch']


        #region parse the provided class file
        try:
            with open(csv_class_file, 'r', newline='') as file:
                self.classes = _read_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(csv_class_file, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key
        #endregion parse the provided class file

        self.image_names = [l for l in self.data_manager.labels]

        super(SAIL_Generator, self).__init__(shuffle_groups=(True if dataset_type=='train' else False), **kwargs)



    def __len__(self):
        if self.__len__value:
            return self.force_steps_per_epoch
        else:
            return len(self.groups)


    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        return 1.

    def load_image(self, image_index):

        return self.data_manager.get_data(self.image_names[image_index])


    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        uid        = self.image_names[image_index]
        annotations = self.data_manager.get_annotations(uid)
        annotations['labels'] = np.array([self.name_to_label(n) for n in annotations['labels']])
        return annotations

        # annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}
        #
        # for idx, annot in enumerate(self.annotation_data[path]['labels_bboxes']):
        #     annotations['labels'] = np.concatenate((annotations['labels'], [self.name_to_label(annot['class'])]))
        #     annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
        #         float(annot['bbox'][0]),
        #         float(annot['bbox'][1]),
        #         float(annot['bbox'][0] + annot['bbox'][2]),
        #         float(annot['bbox'][1] + annot['bbox'][3]),
        #     ]]))
        #
        # return annotations


class data_manager():
    def __init__(self, annotations_file, dataset_type='train', total_examples_number = None):
        assert ((os.path.exists(annotations_file)) & (
            os.path.isfile(annotations_file))), 'annotations file specified either does not exist or not a file'

        self.dataset_type = dataset_type
        self.annotations_file = annotations_file
        self.base_directory = os.path.dirname(os.path.abspath(self.annotations_file))

        print('reading annotatiuons file...')
        start = datetime.now()
        with open(self.annotations_file, 'rb') as f:
            self.labels = pickle.load(f)
        end = datetime.now()
        print('annotatiuons file read in %fs...' % ((end - start).total_seconds()))

        if total_examples_number is not None:
            if len(self.labels) > total_examples_number:
                self.labels = self.labels[np.random.permutation(len(self.labels))[:total_examples_number]]

        #convert it to dictionary for fast access by uid
        self.labels = {l['uid']: l for l in self.labels}

        self.data_files_handlers = {}
        self.mask_files_handlers = {}
        self.segmaps_files_handlers = {}
        if self.dataset_type == 'train':
            self.framemask_files_handlers = {}
        else:
            self.framemask_files_handlers = None

    def get_data(self, uid):
        curr_label = self.labels[uid]
        data_fname_base = curr_label['data_fname_base']
        curr_data_filename = os.path.join(self.base_directory, 'data_%s.npy' % data_fname_base)
        if data_fname_base not in self.data_files_handlers.keys():
            self.data_files_handlers[data_fname_base] = np.load(curr_data_filename, mmap_mode='r')

        data = None
        try:
            data = self.data_files_handlers[data_fname_base][curr_label['transformed_data_idx']]
        except:
            raise (ValueError('error reading data file:\n%s' % curr_data_filename))

        # subset the crop
        x1, y1, x2, y2 = curr_label['crop_bbox']  # (4,) in x1y1x2y2 format
        example_data = data[y1:y2, x1:x2]

        # images are transformed during the training so let them be images-like objects
        return example_data

    def get_all_data(self, uid):
        curr_label = self.labels[uid]
        data_fname_base = curr_label['data_fname_base']
        curr_data_filename = os.path.join(self.base_directory, 'data_%s.npy' % data_fname_base)
        curr_masks_filename = os.path.join(self.base_directory, 'masks_%s.npy' % data_fname_base)
        curr_framemasks_filename = os.path.join(self.base_directory, 'framemask_%s.npy' % data_fname_base)
        curr_segmaps_filename = os.path.join(self.base_directory, 'segmaps_%s.npy' % data_fname_base)

        if data_fname_base not in self.data_files_handlers.keys():
            self.data_files_handlers[data_fname_base] = np.load(curr_data_filename, mmap_mode='r')

        if data_fname_base not in self.mask_files_handlers.keys():
            self.mask_files_handlers[data_fname_base] = np.load(curr_masks_filename, mmap_mode='r')

        if self.dataset_type == 'train':
            if data_fname_base not in self.framemask_files_handlers.keys():
                self.framemask_files_handlers[data_fname_base] = np.load(curr_framemasks_filename, mmap_mode='r')

        if data_fname_base not in self.segmaps_files_handlers.keys():
            self.segmaps_files_handlers[data_fname_base] = np.load(curr_segmaps_filename, mmap_mode='r')

        data = None
        framemask = None
        mask = None
        segmap = None
        try:
            data = self.data_files_handlers[data_fname_base][curr_label['transformed_data_idx']]
            mask = self.mask_files_handlers[data_fname_base][curr_label['transformed_data_idx']]
            if self.dataset_type == 'train':
                framemask = self.framemask_files_handlers[data_fname_base][curr_label['transformed_data_idx']]
            segmap = self.segmaps_files_handlers[data_fname_base][curr_label['transformed_data_idx']]
        except:
            raise (ValueError(
                'error reading one of the following files:\n%s' % str([curr_data_filename,
                                                                       curr_masks_filename,
                                                                       curr_framemasks_filename,
                                                                       curr_segmaps_filename])))

        # subset the crop
        x1, y1, x2, y2 = curr_label['crop_bbox']  # (4,) in x1y1x2y2 format
        example_data = data[y1:y2, x1:x2]
        example_mask = mask[y1:y2, x1:x2]
        if self.dataset_type == 'train':
            example_framemask = framemask[y1:y2, x1:x2]
        else:
            example_framemask = None
        example_segmap = segmap[y1:y2, x1:x2]

        # images are transformed during the training so let them be images-like objects
        return example_data, example_mask, example_framemask, example_segmap

    def get_annotations(self, uid):
        curr_label = self.labels[uid]

        labels = np.array([l['class'] for l in curr_label['labels_bboxes']])
        bboxes = np.array([l['bbox'] for l in curr_label['labels_bboxes']])
        # specific for this retinanet environment
        bboxes = bboxes.astype(np.float64)

        annotations = {'labels': labels, 'bboxes': bboxes}

        return annotations
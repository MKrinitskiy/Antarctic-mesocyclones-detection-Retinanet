import numpy as np
import os, sys
import cv2
from keras_retinanet.preprocessing.generator import Generator
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.utils.data_utils import Sequence
from  datetime import datetime
import pickle
import itertools


# sys.path.append('../')
from utils import *


def sat_label(fname):
    base_filname = os.path.basename(fname)
    reex = '.+(MSG\d).+(\d{14})\.nc'
    match = re.match(reex, base_filname)
    sat_label = match.groups()[0]
    return sat_label



class SAIL_inference_datagenerator:
    def __init__(self, base_data_path = './src_data/', interpolation_constants_directory='./.cache/'):
        base_data_path = os.path.abspath(base_data_path)
        assert os.path.exists(base_data_path), "unable to find base data directory:\n\t%s" % base_data_path
        # assert os.path.isdir(base_data_dir), "base data directory is not a directory:\n\t" % base_data_dir
        assert os.path.exists(interpolation_constants_directory), 'couldn`t find the directory containing pre-calaulated interpolation weights and masks:\n\t' % interpolation_constants_directory

        self.base_data_path = base_data_path
        if os.path.isdir(self.base_data_path):
            self.basepath_type = 'directory'
            data_fnames = find_files(self.base_data_path, '*.nc')

            print('filtering MSG1 data files: from %d' % len(data_fnames))
            data_fnames = [fn for fn in data_fnames if sat_label(fn) != 'MSG1']
            print('filtering MSG1 data files: to %d' % len(data_fnames))

            data_fnames.sort(key=lambda s: s[-17:-3])
            self.data_fnames = data_fnames
        else:
            self.basepath_type = 'file'
            self.data_fnames = [self.base_data_path]

        self.interpolation_constants_directory = interpolation_constants_directory
        # self.base_data_dir = base_data_dir

        self.current = 0

        self.interpolation_constants_memcached = {}
        self.shared_mask_memcached = {}

        # super(SAIL_inference_datagenerator, self).__init__()


    def __len__(self):
        return len(self.data_fnames)


    def __next__(self):
        curr_fname = self.data_fnames[self.current]

        data_lats, data_lons, ch5, ch9, btd, sat_label, dt_str = read_ncfile_data(curr_fname)

        # 20160621140012
        dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")

        interpolation_constants_fname = os.path.join(self.interpolation_constants_directory, 'interpolation_constants_%s.pkl' % sat_label)
        shared_mask_fname = os.path.join(self.interpolation_constants_directory, 'shared_mask_%s.npy' % sat_label)

        if sat_label in self.interpolation_constants_memcached.keys():
            interpolation_constants_dict = self.interpolation_constants_memcached[sat_label]
        else:
            try:
                with open(interpolation_constants_fname, 'rb') as f:
                    interpolation_constants_dict = pickle.load(f)
                    self.interpolation_constants_memcached[sat_label] = interpolation_constants_dict
            except:
                print('unable to read interpolation constants from file %s' % interpolation_constants_fname)
                return None

        interpolation_inds = interpolation_constants_dict['interpolation_inds']
        interpolation_wghts = interpolation_constants_dict['interpolation_wghts']
        interpolation_shape = interpolation_constants_dict['interpolation_shape']

        if sat_label in self.shared_mask_memcached.keys():
            shared_mask = self.shared_mask_memcached[sat_label]
        else:
            try:
                shared_mask = np.load(shared_mask_fname)
                self.shared_mask_memcached[sat_label] = shared_mask
            except:
                print('unable to read shared mask from file %s' % shared_mask_fname)
                return None

        ch5_interpolated = interpolate_data(ch5, interpolation_inds, interpolation_wghts, interpolation_shape)
        ch5_interpolated_ma = np.ma.asarray(ch5_interpolated)
        ch5_interpolated_ma.mask = shared_mask
        ch5_interpolated_ma.data[np.isnan(ch5_interpolated)] = 0.
        ch5_interpolated_ma.mask[np.isnan(ch5_interpolated)] = True
        ch5_interpolated_ma_normed = scale_ch5(ch5_interpolated_ma)
        ch5_interpolated_ma_normed = np.expand_dims(ch5_interpolated_ma_normed, -1)
        ch5_interpolated_ma_normed = np.expand_dims(ch5_interpolated_ma_normed, 0)

        ch9_interpolated = interpolate_data(ch9, interpolation_inds, interpolation_wghts, interpolation_shape)
        ch9_interpolated_ma = np.ma.asarray(ch9_interpolated)
        ch9_interpolated_ma.mask = shared_mask
        ch9_interpolated_ma.data[np.isnan(ch9_interpolated)] = 0.
        ch9_interpolated_ma.mask[np.isnan(ch9_interpolated)] = True
        ch9_interpolated_ma_normed = scale_ch9(ch9_interpolated_ma)
        ch9_interpolated_ma_normed = np.expand_dims(ch9_interpolated_ma_normed, -1)
        ch9_interpolated_ma_normed = np.expand_dims(ch9_interpolated_ma_normed, 0)

        btd_interpolated = interpolate_data(btd, interpolation_inds, interpolation_wghts, interpolation_shape)
        btd_interpolated_ma = np.ma.asarray(btd_interpolated)
        btd_interpolated_ma.mask = shared_mask
        btd_interpolated_ma.data[np.isnan(btd_interpolated)] = 0.
        btd_interpolated_ma.mask[np.isnan(btd_interpolated)] = True
        btd_interpolated_ma_normed = scale_btd(btd_interpolated_ma)
        btd_interpolated_ma_normed = np.expand_dims(btd_interpolated_ma_normed, -1)
        btd_interpolated_ma_normed = np.expand_dims(btd_interpolated_ma_normed, 0)

        example = np.concatenate([ch5_interpolated_ma_normed, ch9_interpolated_ma_normed, btd_interpolated_ma_normed], axis=-1)

        rights = []
        for right in np.arange(512, example.shape[2]+256, 256, dtype=int):
            if right > example.shape[2]:
                right = example.shape[2]
            rights.append(right)
        lefts = np.array(rights) - 512

        tops = []
        for top in np.arange(512, example.shape[1]+256, 256, dtype=int):
            if top > example.shape[1]:
                top = example.shape[1]
            tops.append(top)
        bottoms = np.array(tops) - 512

        crop_bboxes = []
        for (l,b) in itertools.product(lefts, bottoms):
            r = l+512
            t = b+512
            crop_bboxes.append(np.array([l, b, r, t]))

        crops = []
        for crop_bbox in crop_bboxes:
            l, b, r, t = crop_bbox
            curr_crop = example[:, b:t , l:r ,:]
            crops.append(curr_crop)
        crops = np.concatenate(crops, axis=0)


        self.current += 1
        return example,shared_mask,crops,None,curr_fname,dt,crop_bboxes,sat_label
import numpy as np

def scale_btd(data):
    data_scaled_01 = (data+80.)/(5.5+80.)
    data_scaled01_inv = 1.001-data_scaled_01
    data_scaled01_inv_log = np.log(data_scaled01_inv)
    data_scaled2 = 1. - (data_scaled01_inv_log - np.log(0.001))/(-np.log(0.001))
    return data_scaled2

def scale_btd_back(data):
    data_scaled01_inv_log = (1. - data)*(-np.log(0.001)) + np.log(0.001)
    data_exp = np.exp(data_scaled01_inv_log)
    data_exp_inv = 1.001-data_exp
    data_descaled = data_exp_inv*(5.5+80.) - 80.
    return data_descaled


def scale_ch5(data):
    ch5_vmin = 205.
    ch5_vmax = 260.
    return 1 + (ch5_vmin - data)/(ch5_vmax-ch5_vmin)

def scale_ch5_back(data):
    ch5_vmin = 205.
    ch5_vmax = 260.
    return ch5_vmin - (data-1.)*(ch5_vmax-ch5_vmin)


def scale_ch9(data):
    return np.minimum(1. + (200. - data)/(320.-200.), 1.)

def scale_ch9_back(data):
    return 200. - (data-1.)*(320.-200.)

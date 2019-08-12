from .sat_values import *
import numpy as np

def t_brightness_calculate(data, sat_values, channelname = 'ch9'):
    data.mask[data == data.min()] = True
    A = sat_values.A_values()[channelname]
    B = sat_values.B_values()[channelname]
    nu = sat_values.nu_central()[channelname]
    c = sat_values.C2 * nu
    e = nu * nu * nu * sat_values.C1
    logval = np.log(1. + e / data)
    bt = (c / logval - B) / A
    return bt

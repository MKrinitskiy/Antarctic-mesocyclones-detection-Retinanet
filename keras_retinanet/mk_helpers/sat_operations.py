from .sat_values import *
import numpy as np

def t_brightness_calculate(data, channelname = 'ch9'):
    data.mask[data == data.min()] = True
    A = A_values[channelname]
    B = B_values[channelname]
    nu = nu_central[channelname]
    c = C2 * nu
    e = nu * nu * nu * C1
    logval = np.log(1. + e / data)
    bt = (c / logval - B) / A
    return bt

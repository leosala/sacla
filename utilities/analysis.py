import numpy as np
import math
from cython_utils import per_pixel_correction_cython

def print_leaf(f, leaf_name, level=99, init_level=0):
    """
    Print iteratively leafs of an HDF5 file
    """
    try:
        new_leafs = f[leaf_name].keys()
        init_level += 1
        for k in new_leafs:
            if level >= init_level:
                print_leaf(f, leaf_name + "/" + k, level, init_level)
            else:
                print leaf_name + "/" + k
    except:
        try:
            if f[leaf_name].shape[0] > 10:
                print leaf_name, f[leaf_name].shape
            else:
                print leaf_name, f[leaf_name].value
        except:
                print leaf_name, f[leaf_name].value


def rebin(a, *args):
    """
    rebin a numpy array
    """
    shape = a.shape
    lenShape = len(shape)
    factor = np.asarray(shape) / np.asarray(args)
    print factor
    evList = ['a.reshape('] + ['args[%d],factor[%d],' % (i, i) for i in range(lenShape)] + [')'] + ['.mean(%d)' % (i + 1) for i in range(lenShape)]
    return eval(''.join(evList))


def per_pixel_correction_chunked(data, thr, chk_size=300):
    """
    Determins the zero value pixel-by-pixel, using (w x w) pixel regions, and returns a map
    with the mean shift requested. A parameter thr is requested to determine in which
    energy/counts range perform the evaluation.
    """
    tot = data.shape[0]
    result = None
    
    for i in xrange(0, tot, chk_size):
        data_chk = data[i:i + chk_size]
        m = np.ma.masked_where(data_chk >= thr, data_chk)
        m.set_fill_value(0)
        if result is None:
            result = m.sum(axis=0)
        else:
            result += m.sum(axis=0)
    print result.shape
    return result / tot


def per_pixel_correction(data, thr, chk_size=100):
    """
    Determins the zero value pixel-by-pixel, using (w x w) pixel regions, and returns a map
    with the mean shift requested. A parameter thr is requested to determine in which
    energy/counts range perform the evaluation. 
    """
    tot = data.shape[0]
    result = None
    
    for i in xrange(0, tot, chk_size):
        data_chk = data[i:i + chk_size]
        if result is None:
            result = per_pixel_correction_cython(data_chk, thr)
        else:
            result += per_pixel_correction_cython(data_chk, thr)
    print result.shape
    return result / tot


#def per_pixel_correction(data, thr):
#    """
#    Determins the zero value pixel-by-pixel, using (w x w) pixel regions, and returns a map
#    with the mean shift requested. A parameter thr is requested to determine in which
#    energy/counts range perform the evaluation.
#    """
#
#    m = np.ma.masked_where(data > thr, data)
#    m.set_fill_value(0)
#    print m.shape, m, m.mask
#    return np.ma.mean(data, axis=0)


def get_energy_from_theta(thetaPosition):
    # Todo: Most probably these variables need to be read out from the control system ...
    theta_coeff = 25000  # in [pulse/mm]
    lSinbar = 275.0  # in [mm]
    theta_offset = -15.0053431  # in [degree]
    dd = 6.270832

    theta = math.asin((thetaPosition / theta_coeff) / lSinbar + math.sin(theta_offset * math.pi / 180.0)) * 180.0 / math.pi - theta_offset
    energy = 12.3984 / ((dd) * math.sin(theta * math.pi / 180.0))

    return energy

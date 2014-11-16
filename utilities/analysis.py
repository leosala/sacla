import numpy as np
import math
import cython_utils


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
        #print i
        data_chk = data[i:i + chk_size]
        if result is None:
            result = cython_utils.per_pixel_correction_cython(data_chk, thr)
        else:
            result += cython_utils.per_pixel_correction_cython(data_chk, thr)
    print result.shape
    return result / tot


def per_pixel_correction_sacla(h5_dst, tags_list, thr):
    return cython_utils.per_pixel_correction_sacla(h5_dst=h5_dst, tags_list=tags_list, thr=thr)


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


def get_spectrum_sacla(h5_dst, tags_list, corr=None, apply_corr=False, roi=[], masks=[]):
    cython_utils.get_spectrum_sacla(h5_dst, tags_list, corr=corr, apply_corr=apply_corr, roi=roi, masks=masks)


def get_spectrum(data, f="sum", corr=None, chk_size=200, roi=None, masks=[]):
    """
    if masks is a list of lists, then it creates more than one spectrum, one per each mask list
    """
    tot = data.shape[0]
    result = None
    
    masks_tmp = np.array(masks)
    # is it a list of lists?
    if len(masks_tmp.shape) == 3:
        masks_np = masks_tmp
    else:
        masks_np = masks_tmp[np.newaxis, ]
    
    spectra = []

    for masks_list in masks_np:
        total_mask = np.ones(data.shape[0], dtype=bool)
        print "mask", total_mask.shape, masks_list.shape
        if masks != []:
            total_mask = masks_list[0].copy()
            for m in range(1, len(masks)):
                total_mask *= masks_list[m]
        print "mask", total_mask.shape
        for i in xrange(0, tot, chk_size):
            mask = total_mask[i:i + chk_size]
            if roi is not None:
                #print mask.shape, data.shape
                data_chk = data[i:i + chk_size, roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]][mask]
                if corr is not None:
                    data_chk = data_chk - corr[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
            else:
                data_chk = data[i:i + chk_size][mask]
                if corr is not None:
                    data_chk = data_chk - corr

            data_chk = np.nan_to_num(data_chk)
            #print data_chk.shape
            if f == "sum":
                tmp = data_chk.sum(axis=0)
                if result is None:
                    result = tmp
                else:
                    result += tmp

        if f == "mean":
            spectra.append(result.mean(axis=0))
        else:
            spectra.append(result.sum(axis=1))

    if len(spectra) == 1:
        return spectra[0]
    else:
        return spectra

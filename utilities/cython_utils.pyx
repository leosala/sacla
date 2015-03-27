import sys
from time import time
import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool
from libc.math cimport sqrt

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
DTYPE3 = np.double
ctypedef np.double_t DTYPE3_t
DTYPE2 = np.int32
ctypedef np.int32_t DTYPE2_t
DTYPEB = bool
ctypedef bool DTYPEB_t
DTYPE4 = np.int64
ctypedef np.int64_t DTYPE4_t

@cython.boundscheck(False)
@cython.wraparound(False)
def per_pixel_correction_cython(np.ndarray[DTYPE_t, ndim=3] data, float thr):
    """
    Determins the zero value pixel-by-pixel, using (w x w) pixel regions, and returns a map
    with the mean shift requested. A parameter thr is requested to determine in which
    energy/counts range perform the evaluation.
    """

    cdef int tot = data.shape[0]
    cdef int x = data.shape[1]
    cdef int y = data.shape[2]
    cdef np.ndarray[DTYPE_t, ndim = 2] result = np.zeros([x, y], dtype=DTYPE)
    cdef unsigned int i, j, n

    for n in range(tot):
        for i in xrange(0, x):
            for j in xrange(0, y):
                if data[n, i, j] < thr:
                    result[i, j] += data[n, i, j]
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def per_pixel_correction_sacla(h5_dst, np.ndarray[DTYPE4_t, ndim=1] tags_list, int thr, int first_tag, bool get_std=False):

    cdef int x = h5_dst["tag_" + str(first_tag) + "/detector_data"].shape[0]
    cdef int y = h5_dst["tag_" + str(first_tag) + "/detector_data"].shape[1]
    cdef int i = 0
    cdef int tot = tags_list.shape[0]
    cdef np.ndarray[DTYPE_t, ndim = 2] corr_data = np.zeros([x, y], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 2] data = np.zeros([x, y], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 2] std_dev = np.zeros([x, y], dtype=DTYPE)

    for t in tags_list:
        try:
            data = h5_dst["tag_" + str(t) + "/detector_data"][:]
            for i in xrange(0, x):
                for j in xrange(0, y):
                    if data[i, j] < thr:
                        corr_data[i, j] += data[i, j] / tot
                    if get_std:
                        std_dev[i, j] += data[i, j] * data[i, j]   
        except:
            msg = "Tag #%s not found" % t
    if get_std:
        for i in xrange(0, x):
            for j in xrange(0, y):
                std_dev[i, j] = sqrt(std_dev[i, j] / tot - corr_data[i, j] * corr_data[i, j])
        #return corr_data, sqrt(std_dev / tot - corr_data * corr_data)
        return corr_data, std_dev
    else:
        return corr_data

    #if get_std:
    #    return corr_data, sqrt(std_dev / tot - corr_data * corr_data)
    #else:
    #    return corr_data


## @cython.boundscheck(False)
## @cython.wraparound(False)
## def per_pixel_correction_sacla(h5_dst, np.ndarray[DTYPE2_t, ndim=1] tags_list, int thr, int first_tag, bool get_std=False):

##     cdef int x = h5_dst["tag_" + str(first_tag) + "/detector_data"].shape[0]
##     cdef int y = h5_dst["tag_" + str(first_tag) + "/detector_data"].shape[1]
##     cdef int i = 0
##     cdef int tot = tags_list.shape[0]
##     cdef np.ndarray[DTYPE_t, ndim = 2] corr_data = np.zeros([x, y], dtype=DTYPE)
##     cdef np.ndarray[DTYPE_t, ndim = 2] data = np.zeros([x, y], dtype=DTYPE)
##     cdef np.ndarray[DTYPE_t, ndim = 2] std_dev = np.zeros([x, y], dtype=DTYPE)

##     for t in tags_list:
##         try:
##             data = h5_dst["tag_" + str(t) + "/detector_data"][:]
##             for i in xrange(0, x):
##                 for j in xrange(0, y):
##                     if data[i, j] < thr:
##                         corr_data[i, j] += data[i, j] / tot
##                         if get_std:
##                             std_dev[i, j] += data[i, j] * data[i, j]  
##                             #std_dev[i, j] += (data[i, j] * data[i, j]) / tot
##                             #tmp[i, j] += (data[i, j] / tot) * (data[i, j] / tot)
##         except:
##             msg = "Tag #%s not found" % t
    

##     if get_std:
##         #for i in xrange(0, x):
##         #    for j in xrange(0, y):
##         #        std_dev[i, j] = sqrt(std_dev[i, j] - corr_data[i, j] * corr_data[i, j])
##         return corr_data, sqrt(std_dev / tot - corr_data * corr_data)
##         #return corr_data, std_dev
##     else:
##         return corr_data


@cython.boundscheck(False)
@cython.wraparound(False)
def get_spectrum_sacla(h5_dst, np.ndarray[DTYPE4_t, ndim=1] tags_list, DTYPE2_t first_tag, np.ndarray[DTYPE_t, ndim=2] corr=None, roi=[], masks=[], DTYPE_t thr=-9999):

    cdef int x = h5_dst["tag_" + str(first_tag) + "/detector_data"].shape[0]
    cdef int y = h5_dst["tag_" + str(first_tag) + "/detector_data"].shape[1]
    cdef int i = 0
    cdef int tot = tags_list.shape[0]
    cdef np.ndarray[np.uint8_t, cast = True, ndim = 1] total_mask = np.ones(tot, dtype=np.uint8)
    cdef np.ndarray[DTYPE_t, ndim = 2] corr_data
    cdef np.ndarray[DTYPE_t, ndim = 2] data
    cdef int xl
    cdef int xh
    cdef int yl
    cdef int yh
    masks_tmp = np.array(masks)
    # is it a list of lists?
    if len(masks_tmp.shape) == 3:
        masks_np = masks_tmp
    elif len(masks_tmp.shape) == 2:
        masks_np = masks_tmp[np.newaxis, ]
    elif len(masks_tmp.shape) == 1:
        masks_np = masks_tmp[np.newaxis, np.newaxis, ]

    spectra = []

    if roi == []:
        roi = [[0, x], [0, y]]

    xl = roi[0][0]
    xh = roi[0][1]
    yl = roi[1][0]
    yh = roi[1][1]

    corr_data = np.zeros([xh - xl, yh - yl], dtype=DTYPE)
    data = np.zeros([xh - xl, yh - yl], dtype=DTYPE)

    if corr is None:
        corr = np.zeros([x, y], dtype=DTYPE)
    
    for masks_list in masks_np:
        total_mask = np.ones(tot, dtype=DTYPEB)
        corr_data = np.zeros([xh - xl, yh - yl], dtype=DTYPE)
        if masks != []:
            total_mask = masks_list[0].copy()
            if len(masks) == 1:
                break
            for m in range(1, len(masks_list)):
                total_mask *= masks_list[m]

        flag = 0
        for t in tags_list[total_mask]:
            try:
                data = h5_dst["tag_" + str(t) + "/detector_data"][xl:xh, yl:yh] - corr[xl:xh, yl:yh]
                for i in xrange(0, xh - xl):
                    for j in xrange(0, yh - yl):
                        if data[i, j] > thr:
                            corr_data[i, j] += data[i, j]
            except KeyError:
                msg = "Tag " + str(t) + ": cannot find detector data"
            except:
                print sys.exc_info()
        spectra.append([corr_data, corr_data.sum(axis=1)])

    if len(spectra) == 1:
        return spectra[0]
    else:
        return spectra

@cython.boundscheck(False)
@cython.wraparound(False)
def get_spectra_sacla(h5_dst, np.ndarray[DTYPE4_t, ndim=1] tags_list, DTYPE2_t first_tag, np.ndarray[DTYPE_t, ndim=2] corr=None, roi=[], masks=[], DTYPE_t thr=-9999):

    cdef int x = h5_dst["tag_" + str(first_tag) + "/detector_data"].shape[0]
    cdef int y = h5_dst["tag_" + str(first_tag) + "/detector_data"].shape[1]
    cdef int i = 0
    cdef int tot = tags_list.shape[0]
    cdef np.ndarray[np.uint8_t, cast = True, ndim = 1] total_mask = np.ones(tot, dtype=np.uint8)
    cdef np.ndarray[DTYPE_t, ndim = 2] corr_data
    cdef np.ndarray[DTYPE_t, ndim = 2] data
    cdef np.ndarray[DTYPE_t, ndim = 2] total_spectra

    cdef int xl
    cdef int xh
    cdef int yl
    cdef int yh
    masks_tmp = np.array(masks)
    # is it a list of lists?
    if len(masks_tmp.shape) == 3:
        masks_np = masks_tmp
    elif len(masks_tmp.shape) == 2:
        masks_np = masks_tmp[np.newaxis, ]
    elif len(masks_tmp.shape) == 1:
        masks_np = masks_tmp[np.newaxis, np.newaxis, ]

    spectra = []

    if roi == []:
        roi = [[0, x], [0, y]]

    xl = roi[0][0]
    xh = roi[0][1]
    yl = roi[1][0]
    yh = roi[1][1]

    corr_data = np.zeros([xh - xl, yh - yl], dtype=DTYPE)
    data = np.zeros([xh - xl, yh - yl], dtype=DTYPE)

    if corr is None:
        corr = np.zeros([xh - xl, yh - yl], dtype=DTYPE)
    
    for masks_list in masks_np:
        total_mask = np.ones(tot, dtype=DTYPEB)
        corr_data = np.zeros([xh - xl, yh - yl], dtype=DTYPE)
        total_spectra = np.zeros([tot, xh - xl], dtype=DTYPE)

        if masks != []:
            total_mask = masks_list[0].copy()
            if len(masks) == 1:
                break
            for m in range(1, len(masks_list)):
                total_mask *= masks_list[m]

        flag = 0
        for ti, t in enumerate(tags_list[total_mask]):
            try:
                data = h5_dst["tag_" + str(t) + "/detector_data"][xl:xh, yl:yh] - corr  # [xl:xh, yl:yh]
                for i in xrange(0, xh - xl):
                    for j in xrange(0, yh - yl):
                        if data[i, j] > thr:
                            corr_data[i, j] = data[i, j]
                total_spectra[ti, :] = corr_data.sum(axis=1)
            except:
                print "Cython Error", sys.exc_info()
                msg = "Tag " + str(t) + ": cannot find detector data"
        #spectra.append([corr_data, corr_data.sum(axis=1)])

    return total_spectra
    #if len(spectra) == 1:
    #    return spectra[0]
    #else:
    #    return spectra



@cython.boundscheck(False)
@cython.wraparound(False)
def get_roi_data(h5_grp, h5_grp_new, np.ndarray[DTYPE2_t, ndim=1] tags_list, int first_tag, roi, np.ndarray[DTYPE3_t, ndim=2] dark_matrix=None, DTYPE_t pede_thr=-1):
    """
    Writes just  an ROI of original dataset in a new dataset. It assumes a standard SACLA HDF5 internal structure, as: /run_X/detector_Y/tag_Z/detector_data. It also saves: a ROI mask (under h5_grp_new/roi_mask)

    :param h5_grp: source HDF5 group
    :param h5_grp_new: dest HDF5 group
    :param tags_list: list of tags to write
    :param roi: region of interest
    :param dark_matrix: dark correction matrix to be subtracted (optional)
    :param pede_thr: threshold for pedestal computation (optional). If dark_matrix, then the pedestal is computed after the subtraction
    :return: an integer with the actual number of saved tags
    """

    cdef int x = h5_grp["tag_" + str(first_tag) + "/detector_data"].shape[0]
    cdef int y = h5_grp["tag_" + str(first_tag) + "/detector_data"].shape[1]
    cdef int i = 0
    cdef int tot = tags_list.shape[0]
    cdef np.ndarray[np.uint8_t, cast = True, ndim = 2] roi_mask = np.zeros([x, y], dtype=np.uint8)
    cdef int xl = roi[0][0]
    cdef int xh = roi[0][1]
    cdef int yl = roi[1][0]
    cdef int yh = roi[1][1]
    cdef int counter = 0
    cdef np.ndarray[DTYPE_t, ndim = 2] data = np.zeros([xh - xl, yh - yl], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 2] img_sum = np.zeros([x, y], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 2] img_sum2 = np.zeros([x, y], dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim = 2] img = np.zeros([x, y], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 2] corr_data = np.zeros([x, y], dtype=DTYPE)

    for i in xrange(xl, xh):
        for j in xrange(yl, yh):
            roi_mask[i, j] = 1

    if dark_matrix is not None:
        print "computing with dark corr"
    mask_dset = h5_grp_new.create_dataset("roi_mask", data=roi_mask)
    init_time = time()
    for tag in tags_list:
        tag_str = "tag_" + str(tag) + "/detector_data"
        try:
            img = h5_grp[tag_str][:]

            if dark_matrix is not None:
                img -= dark_matrix
            img_sum += img
            img_sum2 += img * img

            data = img[xl:xh, yl:yh]
            grp = h5_grp_new.create_group(h5_grp.name + "/tag_" + str(tag))
            new_dset = h5_grp_new.create_dataset(h5_grp.name + "/" + tag_str, data=data)
            counter += 1

            if pede_thr != -1:
                for i in xrange(0, x):
                    for j in xrange(0, y):
                        if img[i, j] < pede_thr:
                            corr_data[i, j] += img[i, j]
        except KeyError:
            msg = "Tag " + str(tag) + ": cannot find detector data"
        except:
            print sys.exc_info()[0]

        if (100. * float(counter) / float(tot)) % 25 == 0:
            if counter != 0:
                print "%d percent completed" % int(100. * float(counter) / float(tot))
    print "tag loop took ", time() - init_time

    img_sum_dset = h5_grp_new.create_dataset("image_sum", data=img_sum)
    img_avg_dset = h5_grp_new.create_dataset("image_avg", data=img_sum / counter)
    img_std_dset = h5_grp_new.create_dataset("image_std", data=np.sqrt((img_sum2 / counter) - (img_sum / counter) * (img_sum / counter)))

    if dark_matrix is not None:
        dark_dset = h5_grp_new.create_dataset("dark_correction", data=dark_matrix)
    #dset = h5_grp_new.create_dataset("dark_fname", (100,), dtype="S16")
        

    if pede_thr != -1:
        for i in xrange(0, x):
            for j in xrange(0, y):
                corr_data[i, j] /= counter
        pede_dset = h5_grp_new.create_dataset("pedestal_thr" + str(pede_thr), data=corr_data)
        print "pedestal_thr" + str(pede_thr) + " created"

    return counter



@cython.boundscheck(False)
@cython.wraparound(False)
def run_on_images(func, h5_dst, np.ndarray[DTYPE4_t, ndim=1] tags_list, DTYPE2_t first_tag, np.ndarray[DTYPE_t, ndim=2] corr=None, roi=[], masks=[], DTYPE_t thr=-9999):
    """
    It assumes a standard SACLA HDF5 internal structure, as: /run_X/detector_Y/tag_Z/detector_data. 

    :param func:
    :param h5_dst: source HDF5 group
    :param tags_list: list of tags to write
    :param first_tag: 
    :param corr:
    :param roi: region of interest
    :param masks:
    :param pede_thr: threshold for pedestal computation (optional). If dark_matrix, then the pedestal is computed after the subtraction -> does nothing
    :return: an integer with the actual number of saved tags
    """
    cdef int x = h5_dst["tag_" + str(first_tag) + "/detector_data"].shape[0]
    cdef int y = h5_dst["tag_" + str(first_tag) + "/detector_data"].shape[1]
    cdef int i = 0
    cdef int tot = tags_list.shape[0]
    cdef np.ndarray[np.uint8_t, cast = True, ndim = 1] total_mask = np.ones(tot, dtype=np.uint8)
    cdef np.ndarray[DTYPE_t, ndim = 2] corr_data
    cdef np.ndarray[DTYPE_t, ndim = 2] data

    cdef int xl
    cdef int xh
    cdef int yl
    cdef int yh
    masks_tmp = np.array(masks)
    # is it a list of lists?
    if len(masks_tmp.shape) == 3:
        masks_np = masks_tmp
    elif len(masks_tmp.shape) == 2:
        masks_np = masks_tmp[np.newaxis, ]
    elif len(masks_tmp.shape) == 1:
        masks_np = masks_tmp[np.newaxis, np.newaxis, ]

    result = []

    if roi == []:
        roi = [[0, x], [0, y]]

    xl = roi[0][0]
    xh = roi[0][1]
    yl = roi[1][0]
    yh = roi[1][1]

    corr_data = np.zeros([xh - xl, yh - yl], dtype=DTYPE)
    data = np.zeros([xh - xl, yh - yl], dtype=DTYPE)

    if corr is None:
        corr = np.zeros([xh - xl, yh - yl], dtype=DTYPE)
    
    for masks_list in masks_np:
        total_mask = np.ones(tot, dtype=DTYPEB)
        corr_data = np.zeros([xh - xl, yh - yl], dtype=DTYPE)
        if masks != []:
            total_mask = masks_list[0].copy()
            if len(masks) == 1:
                break
            for m in range(1, len(masks_list)):
                total_mask *= masks_list[m]

    #check this... inside or outside the loop???
        flag = 0
        for t in tags_list[total_mask]:
            try:
                data = h5_dst["tag_" + str(t) + "/detector_data"][xl:xh, yl:yh] - corr  # [xl:xh, yl:yh]
            except:
                #print sys.exc_info()
                msg = "Tag " + str(t) + ": cannot find detector data"
            result.append(func(data))

    return result
    

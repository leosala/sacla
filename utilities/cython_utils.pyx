import sys
import numpy as np
cimport numpy as np
cimport cython

from cpython cimport bool

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
DTYPE2 = np.long
ctypedef np.long_t DTYPE2_t
DTYPEB = bool
ctypedef bool DTYPEB_t


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
def get_spectrum_sacla(h5_dst, np.ndarray[DTYPE2_t, ndim=1] tags_list, np.ndarray[DTYPE_t, ndim=2] corr, roi=[], masks=[]):

    cdef int x = h5_dst["tag_" + str(tags_list[0]) + "/detector_data"].shape[0]
    cdef int y = h5_dst["tag_" + str(tags_list[0]) + "/detector_data"].shape[1]
    cdef int i = 0
    cdef int tot = tags_list.shape[0]
    cdef np.ndarray[np.uint8_t, cast = True, ndim = 1] total_mask = np.ones(tot, dtype=np.uint8)
    cdef np.ndarray[DTYPE_t, ndim = 2] corr_data = np.zeros([roi[0][1] - roi[0][0], roi[1][1] - roi[1][0]], dtype=DTYPE)

    masks_tmp = np.array(masks)
    # is it a list of lists?
    if len(masks_tmp.shape) == 3:
        masks_np = masks_tmp
    else:
        masks_np = masks_tmp[np.newaxis, ]
    
    spectra = []

    if roi == []:
        roi = [[0, x], [0, y]]

    for masks_list in masks_np:
    
        total_mask = np.ones(tot, dtype=DTYPEB)

        corr_data = np.zeros([roi[0][1] - roi[0][0], roi[1][1] - roi[1][0]], dtype=DTYPE)
        if masks != []:
            total_mask = masks_list[0].copy()
            print len(masks), 
            if len(masks) == 1:
                break
            for m in range(1, len(masks_list)):
                total_mask *= masks_list[m]

        print tags_list[total_mask].shape
        for i in tags_list[total_mask]:
            try:
                corr_data += h5_dst["tag_" + str(i) + "/detector_data"][roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]] - corr[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
            except:
                msg = "Tag " + str(i) + ": cannot find detector data"
        spectra.append(corr_data.sum(axis=1))

    if len(spectra) == 1:
        return spectra[0]
    else:
        return spectra


@cython.boundscheck(False)
@cython.wraparound(False)
def get_roi_dst(h5_dst, h5_dst_new, fout, np.ndarray[DTYPE2_t, ndim=1] tags_list, roi=[]):

    cdef int x = h5_dst["tag_" + str(tags_list[0]) + "/detector_data"].shape[0]
    cdef int y = h5_dst["tag_" + str(tags_list[0]) + "/detector_data"].shape[1]
    cdef int i = 0
    cdef int tot = tags_list.shape[0]
    cdef np.ndarray[np.uint8_t, cast = True, ndim = 1] total_mask = np.ones(tot, dtype=np.uint8)
    cdef np.ndarray[DTYPE_t, ndim = 2] corr_data = np.zeros([roi[0][1] - roi[0][0], roi[1][1] - roi[1][0]], dtype=DTYPE)

    spectra = []

    if roi == []:
        roi = [[0, x], [0, y]]

    for i in tags_list:
        tag_str = "tag_" + str(i) + "/detector_data"
        try:
            data = h5_dst[tag_str][roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
            grp = fout.create_group(h5_dst.name + "/tag_" + str(i))
            new_dset = fout.create_dataset(h5_dst.name + "/" + tag_str, data=data)
        except:
            msg = "Tag " + str(i) + ": cannot find detector data"
            print sys.exc_info()


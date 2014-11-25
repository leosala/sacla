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
def per_pixel_correction_sacla(h5_dst, np.ndarray[DTYPE2_t, ndim=1] tags_list, int thr, int first_tag):

    cdef int x = h5_dst["tag_" + str(first_tag) + "/detector_data"].shape[0]
    cdef int y = h5_dst["tag_" + str(first_tag) + "/detector_data"].shape[1]
    cdef int i = 0
    cdef int tot = tags_list.shape[0]
    cdef np.ndarray[DTYPE_t, ndim = 2] corr_data = np.zeros([x, y], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 2] data = np.zeros([x, y], dtype=DTYPE)

    for t in tags_list:
        try:
            data = h5_dst["tag_" + str(t) + "/detector_data"][:]
            for i in xrange(0, x):
                for j in xrange(0, y):
                    if data[i, j] < thr:
                        corr_data[i, j] += data[i, j] / tot
        except:
            msg = "Tag #%s not found" % t

    return corr_data


@cython.boundscheck(False)
@cython.wraparound(False)
def get_spectrum_sacla(h5_dst, np.ndarray[DTYPE2_t, ndim=1] tags_list, int first_tag, np.ndarray[DTYPE_t, ndim=2] corr=None, roi=[], masks=[]):

    cdef int x = h5_dst["tag_" + str(first_tag) + "/detector_data"].shape[0]
    cdef int y = h5_dst["tag_" + str(first_tag) + "/detector_data"].shape[1]
    cdef int i = 0
    cdef int tot = tags_list.shape[0]
    cdef np.ndarray[np.uint8_t, cast = True, ndim = 1] total_mask = np.ones(tot, dtype=np.uint8)
    cdef np.ndarray[DTYPE_t, ndim = 2] corr_data = np.zeros([x, y], dtype=DTYPE)

    masks_tmp = np.array(masks)
    # is it a list of lists?
    if len(masks_tmp.shape) == 3:
        masks_np = masks_tmp
    else:
        masks_np = masks_tmp[np.newaxis, ]

    spectra = []

    if roi == []:
        roi = [[0, x], [0, y]]
    else:
        corr_data = np.zeros([roi[0][1] - roi[0][0], roi[1][1] - roi[1][0]], dtype=DTYPE)
    print "ROI", roi
    if corr is None:
        corr = np.zeros([x, y], dtype=DTYPE)

    print corr
    for masks_list in masks_np:
        total_mask = np.ones(tot, dtype=DTYPEB)

        corr_data = np.zeros([roi[0][1] - roi[0][0], roi[1][1] - roi[1][0]], dtype=DTYPE)
        if masks != []:
            total_mask = masks_list[0].copy()
            if len(masks) == 1:
                break
            for m in range(1, len(masks_list)):
                total_mask *= masks_list[m]

        print tags_list[total_mask].shape
        flag = 0
        for i in tags_list[total_mask]:
            try:
                corr_data += h5_dst["tag_" + str(i) + "/detector_data"][roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]] - corr[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
                #print corr_data.shape[0], corr_data.shape[1]
            except:
                msg = "Tag " + str(i) + ": cannot find detector data"

        spectra.append([corr_data, corr_data.sum(axis=1)])

    if len(spectra) == 1:
        return spectra[0]
    else:
        return spectra


@cython.boundscheck(False)
@cython.wraparound(False)
def get_roi_data(h5_grp, h5_grp_new, np.ndarray[DTYPE2_t, ndim=1] tags_list, int first_tag, roi, np.ndarray[DTYPE_t, ndim=2] pede_matrix=None, DTYPE_t pede_thr=-1):
    """
    Writes just  an ROI of original dataset in a new dataset. It assumes a standard SACLA HDF5 internal structure, as: /run_X/detector_Y/tag_Z/detector_data. It also saves: a ROI mask (under h5_grp_new/roi_mask)

    :param h5_grp: source HDF5 group
    :param h5_grp_new: dest HDF5 group
    :param tags_list: list of tags to write
    :param roi: region of interest
    :param pede_matrix: pedestal matrix to be subtracted (optional)
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
    cdef np.ndarray[DTYPE_t, ndim = 2] img = np.zeros([x, y], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 2] corr_data = np.zeros([x, y], dtype=DTYPE)

    for i in xrange(xl, xh):
        for j in xrange(yl, yh):
            roi_mask[i, j] = 1

    mask_dset = h5_grp_new.create_dataset("roi_mask", data=roi_mask)
    for tag in tags_list:
        tag_str = "tag_" + str(tag) + "/detector_data"
        try:
            img = h5_grp[tag_str][:]
            if pede_matrix is not None:
                img -= pede_matrix
            img_sum += img
            data = img[xl:xh, yl:yh]
            grp = h5_grp_new.create_group(h5_grp.name + "/tag_" + str(tag))
            new_dset = h5_grp_new.create_dataset(h5_grp.name + "/" + tag_str, data=data)
            counter += 1

            if pede_thr != -1:
                for i in xrange(0, x):
                    for j in xrange(0, y):
                        if img[i, j] < pede_thr:
                            corr_data[i, j] += img[i, j]
        except:
            msg = "Tag " + str(tag) + ": cannot find detector data"
            # print msg, sys.exc_info()
        if (100. * float(counter) / float(tot)) % 25 == 0:
            print "%d percent completed" % int(100. * float(counter) / float(tot))
    img_sum_dset = h5_grp_new.create_dataset("image_sum", data=img_sum)
    print "a", pede_thr, corr_data
    if pede_matrix is not None:
        pede_dset = h5_grp_new.create_dataset("pedestal", data=pede_matrix)
    if pede_thr != -1:
        for i in xrange(0, x):
            for j in xrange(0, y):
                corr_data[i, j] /= counter
        pede_dset2 = h5_grp_new.create_dataset("pedestal_thr" + str(pede_thr), data=corr_data)
        print "pedestal_thr" + str(pede_thr)

    return counter

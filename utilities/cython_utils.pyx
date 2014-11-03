import numpy as np
cimport numpy as np
cimport cython


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


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

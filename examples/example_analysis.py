# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys


SACLA_LIB = "../"
sys.path.append(SACLA_LIB)
import utilities as ut


def get_line_histos(results, temp, image, axis=0, bins=None):
    """
    This function creates an ADU histogram per each pixel in the direction defined by the axis parameter.
    """
    if bins is None:
        bins = np.arange(-100, 1000, 5)

    for i in range(image.shape[axis]):
        if axis == 0:
            t_histo = np.bincount(np.digitize(image[i, :].flatten(), bins[1:-1]), 
                          minlength=len(bins) - 1)
        elif axis == 1:
            t_histo = np.bincount(np.digitize(image[:, i].flatten(), bins[1:-1]), 
                          minlength=len(bins) - 1)

        if temp["current_entry"] == 0 and i == 0:
            results["histos_adu"] = np.empty([image.shape[axis], t_histo.shape[0]], 
                                                dtype=image.dtype)
        if temp["current_entry"] == 0:           
            results["histos_adu"][i] = t_histo
        else:
            results["histos_adu"][i] += t_histo
    temp["current_entry"] += 1
    return results, temp
    

if __name__ == "__main__":
 
    hf = h5py.File("/home/sala/Work/Data/Sacla/ZnO/257325.h5") 
    dataset1 = "/run_257325/detector_2d_1/"
    roi = [[0, 1024], [0, 250]]
    thr = 65

    an = ut.analysis.AnalysisProcessor()
    an.preprocess_images("set_roi", roi=roi)
    an.preprocess_images("set_thr", thr_low=thr)
        
    an.add_sacla_dataset(hf, dataset1)
    results_spectra = an.add_analysis("spectra", args={'axis': 1, 'thr_low': thr,})
    results_mean = an.add_analysis("mean_std", args={'thr_low': thr})

    bins = np.arange(-150, 300, 5)
    results_histos = an.add_analysis("histos_adu", args={'bins': bins})
    results_histos_line = an.add_analysis(get_line_histos, args={'axis': 0, 'bins': bins})
    results = an.analyze_images()  # n=100)


    plt.figure(figsize=(7, 7))
    plt.plot(results_spectra["spectra"].sum(axis=0), 
             label="ADU > " + str(thr))
    plt.legend(loc='best')
    plt.show()    

    plt.figure(figsize=(7, 7))
    plt.bar(bins[:-1], results_histos["histo_adu"], log=True, width=5)
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.subplot(121)
    plt.imshow(results_histos_line["histos_adu"], vmin=0, vmax=3, 
               extent=(bins[0], bins[-1], roi[0][1], roi[0][0]), aspect=0.5)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(results_mean["mean"], 
               aspect=0.5, vmax=1.5, 
               extent=(roi[1][0], roi[1][1], roi[0][1], roi[0][0]))
    plt.colorbar()
    plt.tight_layout()
    plt.show()    
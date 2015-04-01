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
    if image is None:
        temp["current_entry"] += 1
        return results, temp

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
            results["histos_adu_line"] = np.empty([image.shape[axis], t_histo.shape[0]], 
                                                dtype=image.dtype)
        if temp["current_entry"] == 0:           
            results["histos_adu_line"][i] = t_histo
        else:
            results["histos_adu_line"][i] += t_histo
    temp["current_entry"] += 1
    return results, temp
    

if __name__ == "__main__":
 
    # set filename and dataset name     
    DIR = "/swissfel/photonics/data/2014-11-26_SACLA_ZnO/full_hdf5/"
    #fname = "/home/sala/Work/Data/Sacla/ZnO/258706_roi.h5"
    fname = DIR + "256635-257499/257325.h5"
    dataset_name = "detector_2d_1"
    
    # set up parameters for ROI and threshold
    roi = [[0, 1024], [0, 400]]
    thr = 65

    # create an AnalysisProcessor object
    an = ut.analysis.AnalysisProcessor()
    # if you want a flat dict as a result
    an.flatten_results = True
    
    # add analysis
    an.add_analysis("image_get_spectra", args={'axis': 1, 'thr_low': thr,})
    an.add_analysis("image_get_mean_std", args={'thr_low': thr})
    bins = np.arange(-150, 300, 5)
    an.add_analysis("image_get_histo_adu", args={'bins': bins})
    an.add_analysis(get_line_histos, args={'axis': 0, 'bins': bins})

    # set the dataset
    an.set_sacla_dataset(dataset_name)
    # add preprocess steps
    #an.add_preprocess("image_set_roi", roi=roi)
    #an.add_preprocess("image_set_thr", thr_low=thr)
        
    # run the analysis
    results = an.analyze_images(fname, n=100)

    # plot
    plt.figure(figsize=(7, 7))
    plt.plot(np.nansum(results["spectra"], axis=0), 
             label="ADU > " + str(thr))
    plt.legend(loc='best')
    plt.show()    

    plt.figure(figsize=(7, 7))
    plt.bar(bins[:-1], results["histo_adu"], log=True, width=5)
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.subplot(121)
    plt.imshow(results["histos_adu_line"], vmin=0, vmax=3, 
               extent=(bins[0], bins[-1], roi[0][1], roi[0][0]), aspect=0.5)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(results["mean"], 
               aspect=0.5, #vmax=1.5, 
               extent=(roi[1][0], roi[1][1], roi[0][1], roi[0][0]))
    plt.colorbar()
    plt.tight_layout()
    plt.show()    
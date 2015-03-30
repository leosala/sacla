# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os

SACLA_LIB = "../"
sys.path.append(SACLA_LIB)
print os.environ["PWD"]+"/../"
import utilities as ut

if __name__ == "__main__":
 
    def set_roi(image, roi=[]):
        if roi != []:
            new_image = image[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
            return new_image
        else:
            return image
 
    def set_thr(image, thr_low=0.):
        image[image < thr_low] = 0
        return image
 
    hf = h5py.File("/home/sala/Work/Data/Sacla/ZnO/257325.h5") 
    dataset1 = "/run_257325/detector_2d_1/"
    #roi = [[0, 600], [0, 300]]
    roi = [[0, 1024], [0, 400]] #[460, 512]]
    #roi = [[0, 1024], [0, 512]]
    thr = 65

    an = ut.analysis.AnalysisOnImages()
    an.apply_to_all_images(set_roi, roi=roi)
    #an.apply_to_all_images(set_thr, thr_low=thr)
        
    an.load_sacla_dataset(hf, dataset1)
    an.load_function(ut.analysis.get_spectra, args={'axis': 1, 'thr_low': thr,})
    an.load_function(ut.analysis.get_mean_std, result_f=get_mean_std_results, 
                     args={'thr_low': thr})

    bins = np.arange(-150, 300, 5)
    an.load_function(ut.analysis.get_histo, args={'bins': bins})
    an.load_function(ut.analysis.get_line_histos, args={'axis': 0, 'bins': bins})
    results = an.loop_on_images(n=1000)

    plt.figure(figsize=(7, 7))
    plt.plot(results['get_spectra']["spectra"].sum(axis=0), 
             label="ADU > 0")
    plt.legend(loc='best')
    plt.show()    

    plt.figure(figsize=(7, 7))
    plt.bar(bins[:-1], results["get_histo"]["histo_adu"], log=True, width=5)
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.subplot(121)
    plt.imshow(results["get_line_histos"]["histos_adu"], vmin=0, vmax=3, 
               extent=(bins[0], bins[-1], roi[0][1], roi[0][0]), aspect=0.5)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(results["get_mean_std"]["mean"], 
               aspect=0.5, vmax=1.5, 
               extent=(roi[1][0], roi[1][1], roi[0][1], roi[0][0]))
    plt.colorbar()
    plt.tight_layout()
    plt.show()    
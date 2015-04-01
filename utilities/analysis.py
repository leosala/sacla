import numpy as np
import math
import cython_utils
from time import time
import sys
import h5py
import pydoc

def rebin(a, *args):
    """
    rebin a numpy array
    """
    shape = a.shape
    lenShape = len(shape)
    #factor = np.asarray(shape) / np.asarray(args)
    #print factor
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
    #print result.shape
    return result / tot


def per_pixel_correction_sacla(h5_dst, tags_list, thr, get_std=False):
    first_tag = 0
    for t in h5_dst.keys():
        #print t[0:4]
        if t[0:4] == "tag_":
            first_tag = int(t[4:])
            #print first_tag
            break

    # the astype is needed because the tags can be int32 or in64, depending on the data
    return cython_utils.per_pixel_correction_sacla(h5_dst=h5_dst, tags_list=tags_list.astype(np.int64), thr=thr, first_tag=first_tag, get_std=get_std)


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


def get_spectrum_sacla(h5_dst, tags_list, corr=None, roi=[], masks=[], thr=-9999):
    """
    Loops over SACLA standard HDF5 file, and extracts spectra of images over the Y region. It returns the sum of all the images, and the spectrum obtained. If the mask input is a list of boolean masks, then a list of [sum_of_images, spectrum] is returned.

    :param h5_dst: the HDF5 dataset containing the detector tags, e.g /run_00000/detector_2d_9
    :param tags_list: the list of tags to analyze
    :param corr: per image correction to be applied (tipically a dark image)
    :param masks: a list of selections to apply. Example: [laser_on, laser_off], where laser_on is a boolean mask with the same lenght of tags_list
    :param thr: lower threshold to apply when creating the spectrum and the sum of images, (ADU > thr)

    :returns: sum_of_images, spectrum 
    """
    first_tag = 0
    for t in h5_dst.keys():
        #print t[0:4]
        if t[0:4] == "tag_":
            first_tag = int(t[4:])
            #print first_tag
            break

    return cython_utils.get_spectrum_sacla(h5_dst, tags_list, first_tag, corr=corr, roi=roi, masks=masks, thr=thr)


def get_spectra_sacla(h5_dst, tags_list, corr=None, roi=[], masks=[], thr=-9999):
    first_tag = 0
    for t in h5_dst.keys():
        #print t[0:4]
        if t[0:4] == "tag_":
            first_tag = int(t[4:])
            #print first_tag
            break

    return cython_utils.get_spectra_sacla(h5_dst, tags_list, first_tag, corr=corr, roi=roi, masks=masks, thr=thr)


def run_on_images_sacla(func, h5_dst, tags_list, corr=None, roi=[], masks=[], thr=-9999):
    first_tag = 0
    for t in h5_dst.keys():
        #print t[0:4]
        if t[0:4] == "tag_":
            first_tag = int(t[4:])
            #print first_tag
            break

    return cython_utils.run_on_images(func, h5_dst, tags_list, first_tag, corr=corr, roi=roi, masks=masks, thr=thr)


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


#@profile
def loop_on_images(f, tags_list, dset_name, corr=None, roi=[], spectra_axis=0, create_histos=True, adu_thr=None):
    # variables declaration
    sum_of_counts = []
    spectra = []
    histo = None
    bins = None
    image_sum = None 
    spectra_none = []
    
    for i, t in enumerate(tags_list):
        try:
            image = f[dset_name + "/tag_" + str(t) + "/detector_data"][:]
        except:
            spectra.append(np.nan)
            spectra_none.append(i)
            #sum_of_counts.append(0)
            continue
        # applying corrections
        if corr is not None:
            image -= corr

        # setup ROI
        if roi != []:
            image_roi = image[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
        else:
            image_roi = image

        # from: https://gist.github.com/nkeim/4455635
        if create_histos:
            bins = np.arange(-100, 1000, 5)
            t_histo = np.bincount(np.digitize(image_roi.flatten(), bins[1:-1]), minlength=len(bins) - 1)
            # this is very expensive...
            #t_histo, bins = np.histogram(image_roi, bins=range(-100, 1000, 5))

            if histo is None:
                histo = t_histo
            else:
                histo += t_histo

        if adu_thr is not None:
            image[image < adu_thr] = 0  # np.ma.masked_where(image_t < adu_thr, image_t)
            image_roi[image_roi < adu_thr] = 0
            
        # sum of images
        if image_sum is None:
            image_sum = image
        else:
            image_sum += image
            
        sum_of_counts.append(image_roi.sum())
        spectra.append(image_roi.sum(axis=spectra_axis))

        if corr is None:
            continue

    # correct spectra array for missing tags
    spectra_shape = None
    for i, x in enumerate(spectra):
        if i not in spectra_none:
            spectra_shape = x.shape
            break
    for i in spectra_none:
        #spectra[i] = np.nan * np.zeros(spectra_shape)
        spectra[i] = np.zeros(spectra_shape)
    # remove nans
    spectra = np.array(spectra, dtype=spectra[i].dtype)
    spectra[np.isnan(spectra)] = 0
    return np.array(sum_of_counts, dtype=type(sum_of_counts[0])), histo, bins, image_sum, spectra


#@profile
def get_spectra_from_2D(fname, roi=[], adu_thr=None, corr=False, corr_thr=0., detectors=["detector_2d_1"], spectra_axis=0):
    """
    Get spectra and other useful information from SACLA data files.

    :param fname: hdf5 filename
    :param roi: ROI, as a list, e.g. [[0, 512], [10, 20]]. If a ROI per detector is to be used, it should be a list of lists, in the same order as detectors list. Default: []
    :param adu_thr: lower threshold for ADU counts (not used for spectra and histo computations)
    :param corr: create a Dark correction from this file, using an upper threshold corr_thr. If it is an array, then this array is subtracted to each image
    :param corr_thr: threshold for dark correction computation
    :param detectors: list of 2D detectors to be used in the computation. Default: ["detector_2d_1"]

    :return: a dictionary of results, plus everything contained in the "daq_info/" dataset if available
    """
    results = {}
    t0 = time()
    try:
        f = h5py.File(fname)
        run = f.keys()[-1]
        tags = f[run + "/event_info/tag_number_list"].value
    except:
        print sys.exc_info()
        return {"elapsed_time": -1, "events_s": -1, "source_filename": ""}


    # detectors loop
    for di, detector in enumerate(detectors):
        dset_name = ""
        if detector in f[run].keys():
            dset_name = run + "/" + detector
        else:
            # return a meaningful dict if no images are found
            return {"elapsed_time": -1, "events_s": -1, "source_filename": fname, "run": int(run.replace("run_", ""))}

        # getting the list of tags with images
        tags_list = f[dset_name].keys()[1:]
        # from text to ints
        image_tags_list = [ int(x.replace("tag_", "")) for x in tags_list if x[0:3] == "tag"]

        # getting the spectra
        correction = None
        correction_std = None
        if corr is True:
            correction, correction_std = per_pixel_correction_sacla(f[dset_name + "/"], tags, thr=corr_thr, get_std=True)
        elif isinstance(corr, list):
            correction = corr[di]

        adu_thr2 = adu_thr
        if isinstance(adu_thr, list):
            adu_thr2 = adu_thr[di]

        roi_t = roi
        if len(np.array(roi).shape) == 3:
            roi_t = roi[di]
        
        sum_of_counts, histo, bins, sum_images_noroi, spectra = loop_on_images(f, tags, dset_name, corr=correction, roi=roi_t, spectra_axis=spectra_axis, adu_thr=adu_thr2)

        roi_mask = np.zeros(sum_images_noroi.shape, dtype=int)
        if roi != []:
            #if len(roi) == 2:
            roi_mask[roi_t[0][0]:roi_t[0][1], roi_t[1][0]:roi_t[1][1]] = 1
            #elif len(roi) == 3:
            #    roi_mask[roi[di][0][0]:roi[di][0][1], roi[di][1][0]:roi[di][1][1]] = 1
            #else:
            #    print "[ERROR] Wrong ROI dimension %d, please check" % len(roi)
            #    return {"elapsed_time": -1, "events_s": -1, "source_filename": fname, "run": int(run.replace("run_", ""))}
        else:
            roi_mask[:] = 1


        results[detector + "-images_sum"] = sum_of_counts
        results[detector + "-roi_mask"] = roi_mask
        results[detector + "-sum_image_noroi"] = sum_images_noroi
        results[detector + "-spectra"] = spectra
        results[detector + "-adu_histo"] = histo
        results[detector + "-adu_bins"] = bins
        results[detector + "-dark_corr"] = correction
        results[detector + "-dark_corr_std"] = correction_std

    
    tags_mask = [np.in1d(tags, image_tags_list)]

    #if prefix in 
    #for k, v in f[prefix].iteritems():
    #    tmp_a = v[:][tags_mask]
    #    results[k] = v[:][tags_mask]#np.ndarray(tmp_a.shape, )
    #    #results[k] = tmp_a

        
    te = time() - t0
    results["run"] = int(run.replace("run_", ""))
    results["tags"] = np.array(image_tags_list)
    results["elapsed_time"] = te
    results["events_s"] = len(image_tags_list) / te
    results["is_laser"] = f[run + "/event_info/bl_3/lh_1/laser_pulse_selector_status"][:]
    results["source_filename"] = fname
    return results



def image_get_spectra(results, temp, image_in, axis=0, thr_hi=None, thr_low=None):
    """Returns a spectra (projection) over an axis of an image. This function is to be used within an AnalysisProcessor instance.
    
    Parameters
    ----------
    results : dict
        dictionary containing the results. This is provided by the AnalysisProcessor class
    temp : dict
        dictionary containing temporary variables. This is provided by the AnalysisProcessor class
    image_in : Numpy array
        the image. This is provided by the AnalysisProcessor class
    axis: int, optional
        the axis index over which the projection is taken. This is the same as array.sum(axis=axis). Default is 0
    thr_hi: float, optional
        Upper threshold to be applied to the image. Values higher than thr_hi will be put to 0. Default is None
    thr_low: float, optional
        Lower threshold to be applied to the image. Values lower than thr_low will be put to 0. Default is None
    
    Returns
    -------
    results, temp: dict
        Dictionaries containing results and temporary variables, to be used internally by AnalysisProcessor
    """
    if axis == 1:
        other_axis = 0
    else:
        other_axis = 1

    if temp["current_entry"] == 0:
        results["spectra"] = np.empty((results['n_entries'], temp["image_shape"][other_axis]), dtype=temp["image_dtype"]) 
  
    # if there is no image, return NaN
    if image_in is None:
        results["spectra"][temp['current_entry']] = np.ones(temp["image_shape"][other_axis], dtype=temp["image_dtype"])
        results["spectra"][temp['current_entry']][:] = np.NAN
        temp["current_entry"] += 1
        return results, temp
    
    image = image_in.copy()
    if thr_low is not None:
        image[ image < thr_low] = 0
    if thr_hi is not None:
        image[ image > thr_hi] = 0

    result = image.sum(axis=axis)
          
    results["spectra"][temp['current_entry']] = result
    temp["current_entry"] += 1
    return results, temp

    
def image_get_mean_std(results, temp, image_in, thr_hi=None, thr_low=None):
    """
    later
    """
    if image_in is None:
        return results, temp
        
    image = image_in.copy()
    
    if thr_low is not None:
        image[ image < thr_low] = 0
    if thr_hi is not None:
        image[ image > thr_hi] = 0
    
    if temp["current_entry"] == 0:
        temp["sum"] = np.array(image)
        temp["sum2"] = np.array(image * image)
    else:
        temp["sum"] += image
        temp["sum2"] += np.array(image * image)

    temp["current_entry"] += 1    

    return results, temp    
    

def image_get_mean_std_results(results, temp):
    """
    later
    """
    mean = temp["sum"] / temp["current_entry"]
    std = (temp["sum2"] / temp["current_entry"]) - mean * mean
    std = np.sqrt(std)
    results["mean"] = mean
    results["std"] = std
    return results


def image_get_histo_adu(results, temp, image, bins=None):
    """
    later
    """
    if image is None:
        return results, temp

    if bins is None:
        bins = np.arange(-100, 1000, 5)
    t_histo = np.bincount(np.digitize(image.flatten(), bins[1:-1]), 
                          minlength=len(bins) - 1)
    
    if temp["current_entry"] == 0:
        results["histo_adu"] = t_histo
        results["histo_adu_bins"] = bins
    else:
        results["histo_adu"] += t_histo

    temp["current_entry"] += 1
    return results, temp                  
  

def image_set_roi(image, roi=None):
    """
    Returns a copy of the original image, selected by the ROI region specified
    
    :param image: the input array image
    :param roi: the ROI selection, as [[X_lo, X_hi], [Y_lo, Y_hi]]
    
    :return image: a copy of the original image
    """
    if roi is not None:
        new_image = image[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
        return new_image
    else:
        return image
 
 
def image_set_thr(image, thr_low=None, thr_hi=None, replacement_value=0):
    """
    Apply a low / hi threshold on the image, substituting the thresholded elements with replacement_value
    """
    if thr_low is not None:
        image[image < thr_low] = replacement_value
    if thr_hi is not None:
        image[image > thr_hi] = replacement_value
    return image


class Analysis(object):
    """
    Simple container for the analysis functions to be loaded into AnalysisProcessor. At the moment, it is only
    used internally inside AnalysisProcessor
    """
    def __init__(self, analysis_function, arguments={}, post_analysis_function=None, name=None):
        """
        :param analysis_function: the main analysis function to be run on images
        :param arguments: arguments to analysis_function
        :param post_analysis_function: function to be called only once after the analysis loop
        """

        self.function = analysis_function
        self.post_analysis_function = post_analysis_function
        self.arguments = arguments
        if name is not None:
            self.name = name
        else:
            self.name = self.function.__name__
        self.temp_arguments = {}
        self.results = {}


class AnalysisProcessor(object):
    """
    Simple class to perform analysis on SACLA datafiles. Due to the peculiar file 
    format in use at SACLA (each image is a single dataset), any kind of
    analysis must be performed as a loop over all images: due to this, I/O is
    not optimized, and many useful NumPy methods cannot be easily used.
    
    With this class, each image is read in memory only once, and then passed 
    to the registered methods to be analyzed. All registered functions must:
    + take as arguments at least results (dict), temp (dict), image (2D array)
    + return results, temp
    
    `results` is used to store the results produced by the function, while
    `temp` stores temporary values that must be preserved during the image loop.
    
    A simple example is:

    def get_spectra(results, temp, image, axis=0, ):
        result = image.sum(axis=axis)
        if temp["current_entry"] == 0:
            results["spectra"] = np.empty((results['n_entries'], ) + result.shape, dtype=result.dtype) 
            
        results["spectra"][temp['current_entry']] = result
        temp["current_entry"] += 1
        return results, temp
    
    In order to apply this function to all images, you need to:
    
    # create an AnalysisOnImages object
    an = AnalysisProcessor()

    # load a dataset from a SACLA data file
    fname = "/home/sala/Work/Data/Sacla/ZnO/257325.h5"
    dataset_name = "detector_2d_1"
    an.set_sacla_dataset(hf, dataset_name)

    # register the function:    
    an.add_analysis(get_spectra, args={'axis': 1})

    # run the loop
    results = an.analyze_images(fname, n=1000)

    """

    def __init__(self):
        self.results = []
        self.temp = {}
        self.functions = {}
        self.datasets = []
        self.f_for_all_images = {}
        self.analyses = []
        self.available_analyses = {}
        self.available_analyses["image_get_histo_adu"] = (image_get_histo_adu, None)
        self.available_analyses["image_get_mean_std"] = (image_get_mean_std, image_get_mean_std_results)
        self.available_analyses["image_get_spectra"] = (image_get_spectra, None)
        self.available_preprocess = {}
        self.available_preprocess["image_set_roi"] = image_set_roi
        self.available_preprocess["image_set_thr"] = image_set_thr
        self.n = -1
        self.flatten_results = False

    def __call__(self, dataset_file, dataset_name=None, ):
        #self.set_sacla_dataset(dataset_name)
        return self.analyze_images(dataset_file, n=self.n)

    def add_preprocess(self, f, label="", **kwargs):
        """
        Register a function to be applied to all images, before analysis (e.g. dark subtraction)
        """
        if label != "":
            f_name = label
        elif isinstance(f, str):
            f_name = f
        else:
            f_name = f.__name__

        if isinstance(f, str):
            if not self.available_preprocess.has_key(f):
                raise RuntimeError("Preprocess function %s not available, please check your code" % f)
            self.f_for_all_images[f_name] = {'f': self.available_preprocess[f], "args": kwargs}
        else:
            self.f_for_all_images[f_name] = {'f': f, "args": kwargs}
    
    def list_preprocess(self):
        """List all loaded preprocess functions

        
        Returns
        ----------
        list :
            list of all loaded preprocess functions
        """
        return self.f_for_all_images.keys()
    
    def remove_preprocess(self, label=None):
        """Remove loaded preprocess functions. If called without arguments, it removes all functions.
        
        Parameters
        ----------
        label : string
            label of the preprocess function to be removes
        
        """
        if label is None:
            self.f_for_all_images = {}
        else:
            del self.f_for_all_images[label]
    
    def print_help(self, label=""):
        """Print help for a specific analysis or preprocess function
        
        """
        if label is "":
            print """\n
            ######################## 
            # Preprocess functions #
            ########################\n"""
            for f in self.available_preprocess.values():
                print pydoc.plain(pydoc.render_doc(f))

            print """\n
            ######################## 
            # Analysis  functions  #
            ########################\n"""
            for f in self.available_analyses.values():
                print pydoc.plain(pydoc.render_doc(f[0]))
        else:
            if self.available_preprocess.has_key(label):
                print """\n
                ######################## 
                # Preprocess functions #
                ########################\n"""
                print pydoc.plain(pydoc.render_doc(self.available_preprocess[label]))
            elif self.available_analyses.has_key(label):
                print """\n
                ######################## 
                # Analysis  functions  #
                ########################\n"""
                print pydoc.plain(pydoc.render_doc(self.available_analyses[label][0]))
            else:
                print "Function %s does not exist" % label
                
        
    def add_analysis(self, f, result_f=None, args={}, label=""):
        """
        Register a function to be run on images
        """
        
        if isinstance(f, str):
            if not self.available_analyses.has_key(f):
                raise RuntimeError("Analysis %s not available, please check your code" % f)
            analysis = Analysis(self.available_analyses[f][0], arguments=args, 
                                post_analysis_function=self.available_analyses[f][1], name=f)
        else:
            if label != "":
                analysis = Analysis(f, arguments=args, post_analysis_function=result_f, name=label)
            else:
                analysis = Analysis(f, arguments=args, post_analysis_function=result_f)
        if analysis.name in self.list_analysis():
            print "[INFO] substituting analysis %s" % analysis.name
            self.remove_analysis(label=analysis.name)
        self.analyses.append(analysis)
        return analysis.results

    def list_analysis(self):
        return [x.name for x in self.analyses]

    def remove_analysis(self, label=None):
        if label is None:
            self.analyses = []
        else:
            for an in self.analyses:
                if an.name == label:
                    self.analyses.remove(an)

    def set_sacla_dataset(self, dataset_name, remove_preprocess=True):
        """
        Set the name for the SACLA dataset to be analyzed
        """
        self.dataset_name = dataset_name
        if remove_preprocess:
            print "[INFO] Setting a new dataset, removing stored preprocess functions. To overcome this, use remove_preprocess=False"
            self.remove_preprocess()
        
    def analyze_images(self, fname, n=-1):
        """
        Executes a loop, where the registered functions are applied to all the images
        """
        if self.n != -1 and n == -1:
            n = self.n

        results = {}
        hf = h5py.File(fname, "r")
        self.run = hf.keys()[-1]  # find a better way
        dataset = hf[self.run + "/" + self.dataset_name]
        tags_list = hf[self.run + "/event_info/tag_number_list"].value
        n_images = len(tags_list)
        for analysis in self.analyses:
            analysis.results["n_entries"] = n_images
            analysis.temp_arguments["current_entry"] = 0
        
            # first loop to determine the image size... probably it can be done differently
            for tag in tags_list[0:n]:
                try:
                    image = dataset["tag_" + str(tag) + "/detector_data"][:]
                    analysis.temp_arguments["image_shape"] = image.shape
                    analysis.temp_arguments["image_dtype"] = image.dtype
                    break
                except:
                    pass
            
        #for image in images:
        for tag in tags_list[0:n]:
            try:
                image = dataset["tag_" + str(tag) + "/detector_data"][:]
                if self.f_for_all_images != {}:
                    for k, v in self.f_for_all_images.iteritems():
                        image = v['f'](image, **v['args'])
            except:
                # when an image does not exist, a Nan (not a number) is returned. What to
                # do with this depends on the analysis function itself
                image = None
                    
            for analysis in self.analyses:
                analysis.results, analysis.temporary_arguments = analysis.function(analysis.results, analysis.temp_arguments, image, **analysis.arguments)

        for analysis in self.analyses:
            if analysis.post_analysis_function is not None:
                analysis.results = analysis.post_analysis_function(analysis.results, analysis.temp_arguments)
            if self.flatten_results:
                results.update(analysis.results)
            else:
                results[analysis.name] = analysis.results
        self.results = results
        hf.close()
        return self.results


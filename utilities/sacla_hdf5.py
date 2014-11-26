import h5py
import re
import numpy as np
import datetime
import pytz
import subprocess
import sys

import cython_utils


def get_last_run():
    """Gets the last run from sacla webpage"""

    import urllib2
    # the very important page
    url = "http://xqaccdaq01.daq.xfel.cntl.local/cgi-bin/storage/run.cgi?bl=3"

    # get the html document
    doc = urllib2.urlopen(url).readlines()

    # a bit dirty, but works...
    for i, l in enumerate(doc):
        if l.find("detectors") != -1:
            return int(doc[i + 4].strip().strip("</td>"))


def get_run_metadata(f):
    """
    HDF5 metadata reading: Get run metadata from hdf5 file

    :param f:   Handle to hdf5 file
    :return:    list of runs [{'number':rnumber, 'tags': tags.value, 'startTime':startTime , 'endTime': endTime}, ...]
    """
    keys = f.keys()
    runs = []

    # The following routine assumes following hdf5 structure
    # run_<runNumber> - one for each run
    # run_<runNumber>/event_info/tag_number_list - array holding all tags for specific runs
    # run_<runNumber>/event_info/time_stamp - array holding all timestamps for run (inclusive start time and end time)

    for k in keys:
        # find all runs inside the file
        rmatch = re.match('run_([0-9]+)', k)
        if rmatch:
            # extract run number
            rnumber = rmatch.group(1)
            print 'Matching entry: ' + rnumber

            # find out all tags for run
            tags = f.get(k).get('event_info').get('tag_number_list')

            # find out start and end timestamp of run
            timestamps = f.get(k).get('event_info').get('time_stamp')
            start_time = timestamps.value[0]
            end_time = timestamps.value[-1]

            structure = {'number': rnumber, 'tags': tags.value, 'startTime': start_time, 'endTime': end_time}

            # Code part to return an object instread of a map
            # class Struct(object):
            #    def __init__(self, **entries):
            #        self.__dict__.update(entries)
            #runs.append(Struct(**structure))

            runs.append(structure)
        else:
            print('Skipping entry: ' + k)

    return runs


def syncdaq_get(start_time, end_time, tags, key):
    """
    Read (additional) information via syncdaq_get command

    :param start_time:
    :param end_time:
    :param tags:
    :param key:    value key
    :return:       values of the key for the given start/end time and start/end tag
    """

    start_tag = tags[0]
    end_tag = tags[-1]

    # Example:
    # command = ['syncdaq_get', '-b', '2014-06-12 01:17:18.910107+09:00', '-e', '2014-06-12 01:17:42.871307+09:00', '-f', '219817020', '-a', '219818218', 'xfel_bl_3_st_3_pd_4_fitting_peak/voltage']

    # print 'syncdaq_get', '-b \'', start_time, '\' -e \'', end_time, '\' -f', start_tag, '-a', end_tag, key
    command = ['syncdaq_get', '-b', str(start_time), '-e', str(end_time), '-f', str(start_tag), '-a', str(end_tag), str(key)]

    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    content = proc.stdout.readlines()
    data = []
    data_tags = []
    for l in content:
        l = l.replace('\n', '')

        # strip characters ...
        l = l.replace('not-converged', '0')
        l = l.replace('V', '')
        l = l.replace('saturated', '0')
        l = l.replace('pulse', '')

        if l:
            try:
                (tag, value) = l.split(',')
            except:
                print command
                raise RuntimeError(l)

            # print("%s %s" % (tag.strip(), value.strip()))
            data.append(value.strip())
            data_tags.append(int(tag.strip()))

    # Check consistency of tag list
    if set(tags) != set(data_tags):
        raise RuntimeError('Tag list for key ' + key + ' does not match')

    return data


def get_metadata(runs, variables):
    """
    Get run metadata from the SACLA daq system

    :param runs: List of run(s) metadata
    :param variables: Dict of DAQ variables to get, in the form: {'friendly_name': 'daq_name'}
    :return: Returns all metadata that is specified in variables list. The return structure looks like this: {<run_number>: {'delay':[], 'mono:[]}}
    """

    metadata = {}

    for run in runs:
        meta = {}
        for variable in variables:
            # Call syncdaq_get command (ideally we can retrieve data for all variables at once ...)
            try:
                # print syncdaq_get('2014-06-12 01:17:18.910107+09:00', '2014-06-12 01:17:42.871307+09:00', '219817020', '219818218', variables[variable])

                start_time = datetime.datetime.fromtimestamp(run['startTime'], tz=pytz.timezone('Asia/Tokyo'))
                end_time = datetime.datetime.fromtimestamp(run['endTime'], tz=pytz.timezone('Asia/Tokyo'))

                # to ensure that we are within the tag region by increasing the end/start timestamp by +/- 2 seconds
                start_time = start_time + datetime.timedelta(seconds=-2)
                end_time = end_time + datetime.timedelta(seconds=2)
                meta[variable] = syncdaq_get(start_time, end_time, run['tags'], variables[variable])
            except:
                print sys.exc_info()
                print 'Skipping: ', variable
        metadata[run['number']] = meta

    return metadata


def write_metadata(filename, metadata):
    """

    :param filename:    Name of the hdf5 file to write metadata to
    :param metadata:    Metadata to write to file
    :return:
    """

    out_file = h5py.File(filename, "w")

    for run_number in metadata:
        run = metadata[run_number]

        for variable, values in run.iteritems():
            data_type = np.float
            dat = []
            # if cannot convert to float, cast to NaN
            for d in values:
                try:
                    dat.append(float(d))
                except ValueError:
                    dat.append(np.nan)
            # if variables[variable]["units"] == "bool" or variables[variable]["units"] == "pulse":
            #     data_type = np.int

            print "run_" + str(run_number) + "/daq_info/" + variable
            dataset = out_file.create_dataset("run_" + str(run_number) + "/daq_info/" + variable, data=dat, chunks=True, dtype=data_type)
            # dataset.attrs["units"] = np.string_(variables[variable]["units"])

    out_file.close()


def get_roi_data(h5_dst, h5_dst_new, tags_list, roi, pede_matrix=None, pede_thr=-1):
    """
    Writes just  an ROI of original dataset in a new dataset. It assumes a standard SACLA HDF5 internal structure, as: /run_X/detector_Y/tag_Z/detector_data. It also saves: a ROI mask (under h5_grp_new/roi_mask)

    :param h5_grp: source HDF5 group
    :param h5_grp_new: dest HDF5 group
    :param tags_list: list of tags to write
    :param roi: region of interest
    :param pede_matrix: pedestal matrix to be subtracted (optional)
    :return: an integer with the actual number of saved tags
    """

    first_tag = 0
    for t in h5_dst.keys():
        if t[0:4] == "tag_":
            first_tag = int(t[4:])
            break

    cython_utils.get_roi_data(h5_dst, h5_dst_new, tags_list, first_tag, roi, dark_matrix=pede_matrix, pede_thr=pede_thr)


if __name__ == '__main__':
    # Inputs:
    # hdf5 file - HDF5 file with the specific SACLA structure
    hdf5FileName = '/Users/ebner/Desktop/206178.h5'
    hdf5FileNameMetadata = '/Users/ebner/Desktop/206178_metadata.h5'

    # variables to be read out by 'syncdaq_get' script
    variables = {
        'PD': 'xfel_bl_3_st_3_pd_2_fitting_peak/voltage',
        'PD9': 'xfel_bl_3_st_3_pd_9_fitting_peak/voltage',
        'I0': 'xfel_bl_3_st_3_pd_4_fitting_peak/voltage',
        'M27': 'xfel_bl_3_st_3_motor_27/position',
        'M28': 'xfel_bl_3_st_3_motor_28/position',
        'LaserOn': 'xfel_bl_lh1_shutter_1_open_valid/status',
        'LaserOff': 'xfel_bl_lh1_shutter_1_close_valid/status',
        'Delays': 'xfel_bl_3_st_3_motor_25/position',
        'Mono': 'xfel_bl_3_tc_mono_1_theta/position',
        'APD': 'xfel_bl_3_st_3_pd_14_fitting_peak/voltage',
        'LasI': 'xfel_bl3_st_3_pd_4_peak/voltage',  # Extra info laser I
        'Xshut': 'xfel_bl_3_shutter_1_open_valid/status',  # X-ray on
        'Xstat': 'xfel_mon_bpm_bl3_0_3_beamstatus/summary',  # X-ray status
        'X3': 'xfel_bl_3_st_2_bm_1_pd_peak/voltage',  # X-ray i 3
        'X41': 'xfel_bl_3_st_3_pd_3_fitting_peak/voltage',  # X-ray i 4
        'X42': 'xfel_bl_3_st_3_pd_4_fitting_peak/voltage',  # X-ray i 4
        'Johann': 'xfel_bl_3_st_3_motor_42/position',  # Johann theta
        'APD_trans': 'xfel_bl_3_st3_motor_17/position'  # Johann det
    }

    f = h5py.File(hdf5FileName, 'r')
    runs = get_run_metadata(f)
    metadata = get_metadata(runs, variables)
    write_metadata(hdf5FileNameMetadata, metadata)

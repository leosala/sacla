#!/usr/bin/env python

# This script retrieves all data for runs from a start run number up to the latest one
# If a the data file of a run exist the download will skip this file

# To be executed on some hpc node
# Use: qsub -I to get an interactive shell on a node


data_directory='/work/timbvd/hdf5'
#data_directory='/Users/ebner/Desktop/data'
tmp_data_directory='%s/tmp' % data_directory
taglist_directory='%s/taglists' % data_directory

maketaglist_condition_file='%s/config/maketaglist.conf' % data_directory
dataconvert_config_file='%s/config/dataconvert.conf' % data_directory

beamline=3
run_url = "http://xqaccdaq01.daq.xfel.cntl.local/cgi-bin/storage/run.cgi?bl=3"

import os
import shutil
import time
import sys

def get_last_run():
    """Gets the last run from sacla webpage"""

    import urllib2

    # get the html document
    doc = urllib2.urlopen(run_url).readlines()

    # a bit dirty, but works...
    for i, l in enumerate(doc):
        if l.find("detectors") != -1:
            try:
                return int(doc[i + 4].strip().strip("</td>"))
            except:
                # We return 0 as we are shure that there are no negative runs
                return None

# # FOR TESTING ONLY
# fortest=6
# def get_last_run():
#     global fortest
#     run = fortest
#     if fortest < 20:
#         fortest=fortest+2
#     return run
# # FOR TESTING ONLY

def download_run(current_run, nompccd):
    print current_run
    tmp_data_file = '%s/%06d.h5' % (tmp_data_directory, current_run)
    data_file = '%s/%06d.h5' % (data_directory, current_run)
    if nompccd:
        tmp_data_file = '%s/%06d_nompccd.h5' % (tmp_data_directory, current_run)
        data_file = '%s/%06d_nompccd.h5' % (data_directory, current_run)

    # Check whether run was already downloaded
    if os.path.isfile(data_file):
        print('Datafile for run %s already exists' % current_run)
    else:
        # Start the download of the run (to temporary folder)
        print('Download datafile for run: %s' % current_run)

        # # [begin] DataConvert4
#        os.system("touch %s" % tmp_data_file)
#        time.sleep(1)

        # Make taglist
        #MakeTagList -b 3 -r 243561 -inp FEL_openshutter.txt -out tag_number_run243561.txt
        taglist_file = '%s/%06d_taglist.txt' % (taglist_directory, current_run)
        if nompccd:
            taglist_file = '%s/%06d_taglist_nompccd.txt' % (taglist_directory, current_run)

        command = 'MakeTagList -b %d -r %06d -inp %s -out %s' % (beamline, current_run, maketaglist_condition_file, taglist_file)
        print command
        os.system(command)

        #### Workaround remove empty detector line if exists
        fix_taglist(taglist_file)

        # Call DataConvert4
        # DataConvert4 -f test1017.conf -l tag_number1017.list -dir ./ -o test1017.h5
        command = 'DataConvert4 -f %s -l %s -dir %s -o %06d.h5' % (dataconvert_config_file, taglist_file, tmp_data_directory, current_run)
        if nompccd:
            command = 'DataConvert4 -f %s -l %s -dir %s -o %06d_nompccd.h5' % (dataconvert_config_file, taglist_file, tmp_data_directory, current_run)

        print command
        os.system(command)

        ## [end] DataConvert4

        # Move file to data folder
        print('Move datafile')
        time.sleep(10)  # added by Leo, suspicious crash...
        shutil.move(tmp_data_file, data_file)


def download_run_to_latest(start_run, keepPolling, nompccd):
    if not os.path.exists(tmp_data_directory):
        print 'Create temporary directory %s' % tmp_data_directory
        os.makedirs(tmp_data_directory)

    last_run = get_last_run()
    current_run = start_run
    while last_run is None or current_run <= last_run:
        if last_run is not None:
            while current_run <= last_run:
                try:
                    download_run(current_run, nompccd)
                    current_run += 1
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    print "Failed to get %s because..." % str(current_run)
                    print sys.exc_info()
            print 'checking for new runs'
            last_run = get_last_run()

        if (last_run is None or current_run > last_run) and keepPolling:
            while last_run is None or current_run > last_run:
                print 'no new runs - sleep ...'
                time.sleep(5)
                last_run = get_last_run()


def fix_taglist(txt_file, nompccd):
    import re
    #txt_file='/Users/ebner/Desktop/256664_taglist.txt'
    with open(txt_file, 'r') as file:
        data = file.readlines()
        file.close()

    with open(txt_file, 'w') as file:
        for line in data:
            if re.match('^det,$', line) and not nompccd:
                print 'fix it ...'
            elif re.match('^det,.*', line) and nompccd:
                print 'remove detectors ...'
            else:
                file.write(line)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("run", help="run number")
    parser.add_argument("-l", "--latest", help="download up to latest run number", action="store_true")
    parser.add_argument("-d", "--daemon", help="download up to latest run number and keep polling (only applies if -l is specified)", action="store_true")
    parser.add_argument("-n", "--nompccd", help="Do not include MPCCD detectors", action="store_true")

    arguments = parser.parse_args()

    print arguments.latest
    print arguments.run
    print arguments.daemon
    print arguments.nompccd

    print 'Start run number is "', arguments.run


    if arguments.latest:
        download_run_to_latest(int(arguments.run), arguments.daemon, arguments.nompccd)
    else:
        download_run(int(arguments.run), arguments.nompccd)

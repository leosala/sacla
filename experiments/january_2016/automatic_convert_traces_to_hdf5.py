# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 07:08:55 2016

@author: sala
"""

import os
from convert_traces_to_hdf5 import convert_traces_to_hdf5
from time import sleep, time

indir = "/home/sala/Work/Data/SACLA/Beamtime2014/Streak Data 2/"
sleeptime = 10
time_before_last_run_is_processed = 60  # in seconds

last_run_list = []
last_update = time()
while True:
    try:
        run_list = os.listdir(indir)
        #run_list.sort(key=lambda x: os.path.getmtime(indir + x))
        
        runs = list(set(run_list) - set(last_run_list))
        runs.sort(key=lambda x: os.path.getmtime(indir + x))
        if len(runs) > 1:
            last_update = time()
            last_run_list += runs[:-1]
            for run in runs[:-1]:
                print run
        elif runs != []:
            print time() - last_update > time_before_last_run_is_processed
            if time() - last_update > time_before_last_run_is_processed:
                print "no new directories since %.1f s, processing last directory" % time_before_last_run_is_processed
                print runs[-1]
                last_run_list.append(runs[-1])
        else:
            print "No new directories..."
    except KeyboardInterrupt:
        break
            
    sleep(sleeptime)
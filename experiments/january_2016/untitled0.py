# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 16:35:17 2016

@author: sala
"""

import lecroy
import os
import h5py
import numpy as np

#indir = "/home/sala/Work/Data/SACLA/Beamtime2014/Streak Data/2014-02-18_Run002/"
indir = "/media/cifs/Lcry0715n47498_F/Test_for_h5/"
outdir = "/media/cifs/sencha/test_leo/"
#/media/sshfs/sacla-hpc/work/leonardo/tmp/"

buffer_size = 100  # images

fout = h5py.File(outdir + "test.h5", "w")
groups = {}
bi = 0
li = 0

flist = os.listdir(indir)

traces = ["C1", "C2", "C3", "C4", "F1", "F2"]


for trace in traces:
    print(trace, len(t_flist))

    count = 0
    t_flist = [x for x in flist if x.find(trace) != -1]  

    groups[trace] = fout.create_group(trace)    
    
    t_size = None
    d_size = None

    for fi, f in enumerate(t_flist):
    
        t, d = lecroy.read_timetrace(indir + f)
        if t_size is None:
            t_size = t.shape[0]
            d_size = d.shape[0]
            dset_t = groups[trace].create_dataset("time", (t_size, ), 
                                         compression="gzip", dtype=np.float64)
            dset_d = groups[trace].create_dataset("data", (buffer_size, d_size), maxshape=(None, d_size), 
                                         compression="gzip", dtype=np.float32, 
                                         shuffle=False, compression_opts=6,
                                         chunks=(1, d_size))
        
        if fi == 0:
            spectra_buffer = np.ndarray((buffer_size, d_size), dtype=np.float32)
            bi = 0
            li = 0
        
        if bi < buffer_size:
            spectra_buffer[bi] = d
            bi += 1
        else:
            try:
                dset_d[li:li + bi] = spectra_buffer
                li += bi
                bi = 0
                dset_d.resize(dset_d.shape[0] + buffer_size, axis=0)
            except:
                print(li, bi, spectra_buffer.shape)
        count += 1
        
    print(count)
    dset_t[:] = t
    if dset_t.shape[0] > count:
        dset_d.resize(count, axis=0)
        print(dset_d.shape)
    
       
fout.close()


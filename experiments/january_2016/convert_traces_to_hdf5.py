# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 16:35:17 2016

@author: sala
"""

import lecroy
from read_serial import read_serial
import os
import h5py
import numpy as np
import sys

#indir = "/home/sala/Work/Data/SACLA/Beamtime2016/Test_for_h5/"
indir = "/media/cifs/Lcry0715n47498_F/Test_10000_shots/"
outdir = "/media/cifs/sencha/test_leo/"
#/media/sshfs/sacla-hpc/work/leonardo/tmp/"
#outdir = "/tmp/"


tag_trace = "C4"


def convert_traces_to_hdf5(indir, outdir, outfile=None, tag_trace="C4", test_trace=None):
    g_tags = []
        
    flist = os.listdir(indir)
    # get traces from filename list
    traces = list(set(map(lambda x: x[:2], flist)))
    # tag trace goes first
    traces.remove(tag_trace)
    traces.insert(0, tag_trace)
    
    print "Found this traces: ", traces    
    
    buffer_size = 100  # images
    default_value = 0
    
    if outfile is None:
        outfile = indir.strip("/").split("/")[-1] + ".h5"
        
    fout = h5py.File(outdir + outfile, "w")
    groups = {}
    bi = 0
    li = 0
    
    #fnumbers = sorted(set(map(lambda x: x[7:12], flist)))
    fnumbers = None
    t_flists = {}
    t_fnumbers = {}

    # running only on events with all traces
    for trace in traces:
        t_flists[trace] = filter(lambda x: x.find(trace) != -1, flist)  
        t_fnumbers[trace] = sorted(set(map(lambda x: x[7:12], t_flists[trace])))   
        if fnumbers is None:
            fnumbers = t_fnumbers[trace]
        else:
            fnumbers = set(fnumbers) & set(t_fnumbers[trace])
        
    fnumbers = list(sorted(fnumbers))
    for trace in traces:
        count = 0
        t_flist = t_flists[trace]
        # first, tags... skipping events with no tag
    
        groups[trace] = fout.create_group(trace)    
        
        t_size = None
        d_size = None
        t = None
    
        #for fi, f in enumerate(t_flist):
   
        if fnumbers == [] or fnumbers is None:
            print "No tags found, cannot continue"
            break
        if t_flist == []:
            print "No files found"
            break
        
        for fi, n in enumerate(fnumbers):
            fname = indir + trace + "Trace" + str(n).zfill(5) + ".trc"
            try:
                bwf = lecroy.LecroyBinaryWaveform(fname)
                t = bwf.WAVE_ARRAY_1_time
                d = bwf.WAVE_ARRAY_1.ravel()

                #t, d = lecroy.read_timetrace(indir + trace + "Trace" + str(n).zfill(5) + ".trc")
            except:
                print "[ERROR] file %s" % fname
                print sys.exc_info()[1]
                # at least the first n should exist for all traces
                if t is None:
                    continue
                else:
                    d = np.zeros((d_size, ), dtype=np.float32)
                    #print d
                    d[:] = default_value
                
            if t_size is None:
                t_size = t.shape[0]
                d_size = d.shape[0]
                dset_t = groups[trace].create_dataset("time", (t_size, ), 
                                             compression="gzip", dtype=np.float64)
                dset_d = groups[trace].create_dataset("data", (buffer_size, d_size), maxshape=(None, d_size), 
                                             compression="gzip", dtype=np.float32, 
                                             shuffle=False, compression_opts=6,
                                             chunks=(1, d_size))
                dset_n = groups[trace].create_dataset("filenumber", (buffer_size, ), maxshape=(None, ),
                                             dtype=np.int, chunks=True)
                if trace == tag_trace:
                    dset_tag = fout.create_dataset("tags", (buffer_size, ), maxshape=(None, ),
                                             dtype=np.int64, chunks=True)
            
            if fi == 0:
                spectra_buffer = np.ndarray((buffer_size, d_size), dtype=np.float32)
                n_buffer = np.ndarray((buffer_size, ), dtype=np.int)
                if trace == tag_trace:
                    tag_buffer = np.ndarray((buffer_size, ), dtype=np.int64)
                bi = 0
                li = 0
            
            if bi < buffer_size:
                spectra_buffer[bi] = d
                if trace == tag_trace:
                    #print d, bwf.HORIZ_INTERVAL
                    tag_buffer[bi] = read_serial(d, bwf.HORIZ_INTERVAL)[0]
                    g_tags.append(tag_buffer[bi])
                    #print tag_buffer[bi]
                n_buffer[bi] = n
                bi += 1
            else:
                try:
                    dset_d[li:li + bi] = spectra_buffer
                    dset_n[li:li + bi] = n_buffer
                    dset_d.resize(dset_d.shape[0] + buffer_size, axis=0)
                    dset_n.resize(dset_n.shape[0] + buffer_size, axis=0)

                    if trace == tag_trace:
                        dset_tag[li:li + bi] = tag_buffer
                        dset_tag.resize(dset_tag.shape[0] + buffer_size, axis=0)

                    li += bi
                    bi = 0
                except:
                    print sys.exc_info()
                    #print li, bi, spectra_buffer.shape
            count += 1
        
        if t_size is None:
            print "No files found! Are tag traces missing?"
            continue            
        print count
        dset_t[:] = t
        if bi != 0:
            try:
                dset_d[li:li + bi] = spectra_buffer[:bi]
                dset_n[li:li + bi] = n_buffer[:bi]
                #dset_d.resize(dset_d.shape[0] + buffer_size, axis=0)
                #dset_n.resize(dset_n.shape[0] + buffer_size, axis=0)
                
                if trace == tag_trace:
                    dset_tag[li:li + bi] = tag_buffer[:bi]
                    #dset_tag.resize(dset_tag.shape[0] + buffer_size, axis=0)
                li += bi
                count = li
                
                bi = 0
            except:
                print sys.exc_info()
                
        if dset_t.shape[0] >= count:
            dset_d.resize(count, axis=0)
            dset_n.resize(count, axis=0)
            dset_tag.resize(count, axis=0)
           
    fout.close()
    print "Output file %s created" % (outdir + outfile)
    return g_tags, outdir + outfile


if __name__ == "__main__":
    tags, outfile  = convert_traces_to_hdf5(indir, outdir, )
    hf = h5py.File(outfile)
    r_tags = hf['tags'][:]
    hf.close()
    assert( (tags - r_tags == 0).all() )
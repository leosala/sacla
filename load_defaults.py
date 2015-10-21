# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:40:48 2015

@author: sala
"""

import os

PWD = os.getcwd()
TOOLS_DIR = PWD + "/../"

import numpy as np
print "Imported NumPy as np"

import matplotlib.pyplot as plt
plt.style.use('ggplot')
print "Imported Matplotlib.pyplot as plt, with ggplot style"

import h5py
print "Imported h5py"

import sys

# Loading SACLA tools 
SACLA_LIB = PWD
sys.path.append(SACLA_LIB)
import utilities as sacla_utils
print "Imported sacla_utils"

# Loading ImagesProcessor
try:
    from photon_tools.images_processor import ImagesProcessor
    from photon_tools.plot_utilities import plot_utilities as pu
    import photon_tools.hdf5_utilities as h5u
    print "Imported ImagesProcessor"
    print "Imported plot_utilities as pu"
    print "Imported hdf5_utilities as h5u"
except:
    try:
        sys.path.append(TOOLS_DIR)
        from photon_tools.images_processor import ImagesProcessor
        from photon_tools.plot_utilities import plot_utilities as pu
        import photon_tools.hdf5_utilities as h5u
        print "Imported ImagesProcessor"
        print "Imported plot_utilities as pu"
        print "Imported hdf5_utilities as h5u"
    except:
        print "[ERROR] cannot load ImagesProcessor library"

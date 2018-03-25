# Overview
This repository holds all the scripts and ipython notebooks used for the PSI November SACLA beamtime.

Before using go to utilities folder and execute `compile_cython.sh` to compile the cython code!

**Code is provided without any guarantee**

# Installation

## Python environment (Linux)

The suggested way to have a fully functional and updated Python environment is to use the Anaconda Python Distribution (https://store.continuum.io/cshop/anaconda/). Download it, install it in some local directory `<anaconda_dir>`, and then export the new `PATH`:

```
export PATH=<anaconda_dir>/bin:${PATH}'
```

When starting `ipython`, you should then get something like:

```
$ ipython
Python 2.7.9 |Continuum Analytics, Inc.| (default, Mar  9 2015, 16:20:48) 
Type "copyright", "credits" or "license" for more information.

IPython 2.2.0 -- An enhanced Interactive Python.
Anaconda is brought to you by Continuum Analytics.
Please check out: http://continuum.io/thanks and https://binstar.org
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.

In [1]: 
```

## Getting this code

This code needs the *photon_tools* library, located here: (https://github.com/leosala/photon_tools.git).

You can either clone this git repository, e.g.:

```
mkdir <mydir>
cd <mydir>
git clone https://github.com/leosala/photon_tools.git
cd photon_tools
python setup.py develop
cd ..
git clone https://github.com/leosala/sacla.git
```

or just download a tarball, e.g.:

```
wget https://github.com/leosala/photon_tools/archive/master.zip
wget https://github.com/leosala/sacla/archive/master.zip
```

Default libraries (tools, numpy, etc) can be loaded doing `%load load_defaults.py` in an IPython shell.

Please compile the Cython extensions before running the code (this will likely disappear...)

```
cd utilities
./compile_cython.sh
```

# Examples

## Exploring HDF5 files




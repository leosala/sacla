import h5py


class Struct:
    """
    The recursive class for building and representing objects of an hdf5 tree

    Example:
    --------

    import h5py
    f = h5py.File('../256985.h5', 'r')
    o = Struct(f)

    From now on you can access values like this:

    x.file_info.version
    instead of this:
    f.get('file_info').get('version').value
    """

    def __init__(self, obj):
        for k in obj.keys():
            v = obj.get(k)
            if not isinstance(v, h5py._hl.dataset.Dataset):
                setattr(self, k, Struct(v))
            else:
                setattr(self, k, v.value)

    def __getitem__(self, val):
        return self.__dict__[val]

    def __repr__(self):
        return '{%s}' % str(', '.join('%s : %s' % (k, repr(v)) for
                                      (k, v) in self.__dict__.iteritems()))


class StructSOnly:
    """
    Same as Struct object just that this wrapper holds that type of data instead of the actual value
    """

    def __init__(self, obj):
        import re

        for k, v in obj.iteritems():
            if not isinstance(v, h5py._hl.dataset.Dataset):
                if not re.match('^tag_.*', k):  # Exclude tagged detector images
                    setattr(self, k, StructSOnly(v))
            else:
                setattr(self, k, str(v.dtype))

    def __getitem__(self, val):
        return self.__dict__[val]


def printStructure(x, level):
    for a, b in x.__dict__.iteritems():
        if isinstance(b, StructSOnly):
            print ' ' * level, a
            printStructure(b, level + 4)
        else:
            print ' ' * level, a, '[' + b + ']'


def search_hdf5(hdf5_group, regexp, print_elems=0):
    """
    Simple utility to search for dataset/group names inside an hdf5.
    WARNING: if searching in a group with a lot of subgroups/datasets, can be slow! Please search always within a well defined group
    
    :param hdf5_group:
    :param regexp:
    :param print_elems: print the first X elements, default:0
    """
    #import re
    #rule = re.compile(regexp)
    def visit_f(name, object):
        if name.find(regexp) != -1:
            if print_elems == 0:
                if isinstance(object, h5py.Dataset):
                    print name, object.dtype, object.shape
            else:
                if isinstance(object, h5py.Dataset):
                    print name, object.dtype, object.shape, object[:print_elems]

    hdf5_group.visititems(visit_f)


def print_leaf(f, leaf_name="/", level=0, init_level=0):
    """
    Print iteratively leafs of an HDF5 file

    :param leaf_name: The group or the dataset from where to start
    :param level: Recursiveness level
    :param init_level:
    """

    try:
        new_leafs = f[leaf_name].keys()
        init_level += 1
        for k in new_leafs:
            if level >= init_level:
                print_leaf(f, leaf_name + "/" + k, level, init_level)
            else:
                print (leaf_name + "/" + k).replace("//", "/")
    except:
        try:
            if f[leaf_name].shape[0] > 10:
                print leaf_name, f[leaf_name].shape
            else:
                print leaf_name, f[leaf_name].value
        except:
                print leaf_name, f[leaf_name].value

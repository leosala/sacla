
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


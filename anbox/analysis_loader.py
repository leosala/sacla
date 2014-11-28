import os
import sys
import glob
import plugins


def get_plugins_list():
    plugins_list = []
    # plugins_dir = os.path.abspath("plugins") + "/*.py"
    # print os.path.split(__file__)
    plugins_dir = os.path.join(os.path.split(__file__)[0], "plugins/*.py")
    for f in glob.glob(plugins_dir):
        if os.path.isfile(f) and not os.path.basename(f).startswith('_'):
            plugins_list.append(os.path.basename(f)[:-3])
    return plugins_list


def load(htype):
    try:
        s_type = getattr(plugins, htype)  # __import__("plugins." + htype, fromlist=".")
        modules_list = [i for i in dir(s_type) if i[0] != "_"]
        # assert(len(modules_list) == 1)
        an_plug = getattr(s_type, modules_list[0])  # __import__("plugins." + htype + "." + modules_list[0], fromlist=".")
    except:
        print "Cannot load plugin %s" % htype
        sys.exit(-1)
    return an_plug()


if __name__ == "__main__":
    plugins_list = get_plugins_list()
    print plugins_list
    fname = "/work/timbvd/hdf5/257508.h5"
    plugin_conf = {}
    plugin_conf['create_spectra'] = {"roi": [[0, 1024], [325, 335]]}
    for p in plugins_list:
        print "Loading %s" % p
        algo = load(p)
        algo.apply(fname)
        print algo.run()

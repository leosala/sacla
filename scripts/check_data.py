#!/usr/bin/env python

"""
Check whether all files are downloaded
"""
import h5py
import re


def check_hdf5(file_path, data_files, postfix=""):

    corrupted_files = []
    for file_n in data_files:
        try:
            check_file = h5py.File(file_path+'/'+file_n, 'r')
            if list(check_file.keys()) == {}:
                f_name = re.sub('%s.h5$' % postfix, '', file_n)
                corrupted_files.append(f_name)
            check_file.close()
        except:
            f_name = re.sub('%s.h5$' % postfix, '', file_n)
            corrupted_files.append(f_name)

    # In a very bad case we might have 2 same entries in the list of corrupted files ...
    # As we have to check the file anyway we don't care ...

    return corrupted_files


def check_for_missing_files(data_files, postfix=""):

    data_files.sort()
    current_number = int(data_files[0].replace(postfix + '.h5', ''))

    missing_numbers = []

    # Ensure that afterwards we are not running into an endless loop
    if len(data_files) == 1:
        return missing_numbers

    for v in data_files:
        while not v == '%06d%s.h5' % (current_number, postfix):
            missing_numbers.append(current_number)
            current_number += 1
        current_number += 1
    return missing_numbers


def get_files(file_path, postfix=""):
    from os import listdir
    from os.path import isfile, join
    import re

    # Filter out all files matching the naming convention 000000.h5
    data_files = [f for f in listdir(file_path) if
                  (isfile(join(file_path, f)) and re.match('^[0-9]{6}' + postfix + '.h5$', f))]

    return data_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="directory to check")
    parser.add_argument("-p", "--postfix", help="Optional postfix, e.g. '_nompccd'", action="store", default="")

    arguments = parser.parse_args()

    directory = arguments.directory
    postfix = arguments.postfix

    data_files = get_files(directory, postfix)
    missing_files = check_for_missing_files(data_files, postfix)

    print('Following files are missing ...')
    print(len(missing_files))
    print(missing_files)

    corrupted_files = check_hdf5(directory, data_files, postfix)
    print('Following files are corrupted ...')
    print(len(corrupted_files))
    print(corrupted_files)
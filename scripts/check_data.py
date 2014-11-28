#!/usr/bin/env python

"""
Check whether all files are downloaded
"""


def check_for_missing_files(file_path, postfix=""):
    from os import listdir
    from os.path import isfile, join
    import re

    # Filter out all files matching the naming convention 000000.h5
    data_files = [f for f in listdir(file_path) if (isfile(join(file_path, f)) and re.match('^[0-9]{6}'+postfix+'.h5$', f))]
    current_number = int(data_files[0].replace(postfix+'.h5', ''))

    data_files.sort()

    missing_numbers = []

    # Ensure that afterwards we are not running into an endless loop
    if len(data_files) == 1:
        return missing_numbers

    for v in data_files:
        while not v == '%06d%s.h5' % (current_number,postfix):
            missing_numbers.append(current_number)
            current_number +=1
        current_number += 1
    return missing_numbers


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="directory to check")
    parser.add_argument("-p", "--postfix", help="Optional postfix, e.g. '_nompccd'", action="store", default="")

    arguments = parser.parse_args()

    missing_files = check_for_missing_files(arguments.directory, arguments.postfix)

    print 'Following files are missing ...'
    print missing_files
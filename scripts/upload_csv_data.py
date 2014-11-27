#!/usr/bin/env python

import os

base_url = 'http://10.12.177.227:8888/api/plot/run'


def upload_value(txt_file, run_number):
    csv_file = txt_file
    upload_url = base_url+'/'+run_number

    with open(txt_file, 'r') as file:
        data = file.readlines()
        file.close()

    # with open(csv_file, 'w') as file:
    #     for line in data:
    #         line = line.replace(",\n", "\n")
    #         #line = line.replace("\t", ",")
    #         file.write(line)
    #     file.close()

    import requests

    with open(csv_file) as file:
        response = requests.put(upload_url, data=file.read())
        file.close()

    #os.remove(csv_file)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("file", help="Value files")
    parser.add_argument("run", help="Run number")

    arguments = parser.parse_args()

    print arguments.file
    print arguments.run

    upload_value(arguments.file, arguments.run)


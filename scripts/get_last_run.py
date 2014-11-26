#!/usr/bin/env python

import urllib2


url = "http://xqaccdaq01.daq.xfel.cntl.local/cgi-bin/storage/run.cgi?bl=3"
url = "file:///Users/ebner/Git/sacla/test/test.html"


if __name__ == "__main__":
    doc = urllib2.urlopen(url).readlines()[80:90]

    for i, l in enumerate(doc):
        if l.find("detectors") != -1:
            print i, int(doc[i + 4].strip().strip("</td>"))

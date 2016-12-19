#!/usr/bin/python

__author__ = "Matan Lachmish"
__copyright__ = "Copyright 2016, Tel Aviv University"
__version__ = "1.0"
__status__ = "Development"

import sys
import os
import urllib
import py7D
import hdf5_getters

def die_with_usage():
    """ HELP MENU """
    print 'USAGE: python previewDownloader.py [path to MSD data]'
    sys.exit(0)

def update_progress(progress):
    print '\r[{0}] {1}%'.format('#' * (progress / 10), progress)

# def download

if __name__ == "__main__":
    # help menu
    if len(sys.argv) < 2:
        die_with_usage()

    msdPath = sys.argv[1]
    i = 0.0
    for folder in os.listdir(msdPath):
        insidePath = msdPath+'/'+folder
        if (os.path.isdir(insidePath)):
            for folder2 in os.listdir(insidePath):
                insidePath2 = insidePath + '/' + folder2
                if (os.path.isdir(insidePath2)):
                    for file in os.listdir(insidePath2):
                        previewFilePath = insidePath2 + '/' + os.path.splitext(file)[0] + '.mp3'
                        print previewFilePath
                        if file.endswith('h5') and not os.path.isfile(previewFilePath):
                            h5FilePath = insidePath2+'/'+file
                            # print 'Processing ' + h5FilePath

                            try:
                                h5 = hdf5_getters.open_h5_file_read(h5FilePath)
                                id7Digital = hdf5_getters.get_track_7digitalid(h5)
                                h5.close()

                                url = py7D.preview_url(id7Digital)
                                urlretrieve = urllib.urlretrieve(url, previewFilePath)
                            except Exception as e:
                                print "Error accured: " + str(e)

                        if file.endswith('h5'):
                            # update_progress(int(i/7620 * 100))
                            sys.stdout.write("\r%d%%" % int(i/7620 * 100))
                            sys.stdout.flush()
                            i += 1
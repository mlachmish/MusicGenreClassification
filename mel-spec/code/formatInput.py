#!/usr/bin/python

from __future__ import print_function

__author__ = "Matan Lachmish"
__copyright__ = "Copyright 2016, Tel Aviv University"
__version__ = "1.0"
__status__ = "Development"

import numpy as np
from matplotlib import pyplot as plt
from tsne import bh_sne
import pickle
import sys
import os
from numpy import array

def die_with_usage():
    """ HELP MENU """
    print ('USAGE: python visuallizePreproccess.py [path to MSD pp data]')
    sys.exit(0)

def update_progress(progress):
    print ('\r[{0}] {1}%'.format('#' * (progress / 10), progress))

labelsDict = {
    'blues'     :   0,
    'classical' :   1,
    'country'   :   2,
    'disco'     :   3,
    'hiphop'    :   4,
    'jazz'      :   5,
    'metal'     :   6,
    'pop'       :   7,
    'reggae'    :   8,
    'rock'      :   9,
}

if __name__ == "__main__":

    # help menu
    if len(sys.argv) < 2:
        die_with_usage()

    #Load Data
    data = {}
    labels = []

    i = 0.0
    walk_dir = sys.argv[1]
    print('walk_dir = ' + walk_dir)

    for root, subdirs, files in os.walk(walk_dir):
        for filename in files:
            if filename.endswith('pp'):
                file_path = os.path.join(root, filename)
                # print('\t- file %s (full path: %s)' % (filename, file_path))

                with open(file_path, 'r') as f:
                    try:
                        soundId = os.path.splitext(filename)[0]
                        content = f.read()
                        pp = pickle.loads(content)
                        pp = np.asarray(pp)
                        # pp = np.delete(pp, 1, axis=2)
                        data[soundId] = pp

                        labelName = filename.split('.')[0]
                        labelAsArray = [0] * len(labelsDict)
                        labelAsArray[labelsDict[labelName]] = 1
                        labels.append(labelAsArray)
                    except Exception as e:
                        print ("Error occurred" + str(e))

            if filename.endswith('pp'):
                sys.stdout.write("\r%d%%" % int(i / 1000 * 100))
                sys.stdout.flush()
                i += 1

    #save data and lables
    with open("data", 'w') as f:
        f.write(pickle.dumps(data.values()))

    with open("labels", 'w') as f:
        f.write(pickle.dumps(array(labels)))

    # print sizes
    print("Data set size: " + str(len(data.keys())))
    print("Number of genres: " + str(len(labelsDict.keys())))

    # convert image data to float64 matrix. float64 is need for bh_sne
    reshapedList = array(data.values())
    x_data = np.asarray(reshapedList).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))

    # perform t-SNE embedding
    vis_data = bh_sne(x_data, perplexity=30)

    # plot the result
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    colors = []
    for label in labels:
        colors.append(label.index(1))

    plt.scatter(vis_x, vis_y, c=colors, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10), label='Genres')
    plt.clim(-0.5, 9.5)
    plt.title('t-SNE mel-spectrogram samples as genres')
    plt.show()
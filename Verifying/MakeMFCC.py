#! /usr/bin/python2.7
import os
import scipy.io.wavfile
import numpy as np
import glob
from matplotlib import pylab
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scikits.talkbox.features import mfcc

MEOW_DIR="/home/colin/Documents/Octave/CatMeow/"

positive_training_directory = "WithCat/Training/"
positive_training_files = []
positive_validation_directory = "WithCat/Validation/"
positive_validation_files = []
negative_training_directory = "NoCat/Training/"
negative_training_files = []
negative_validation_directory = "NoCat/Validation/"
negative_validation_files = []

for root, _, files in os.walk(positive_training_directory):
    for f in files:
        if f.endswith("wav"):
            positive_training_files.append(os.path.join(positive_training_directory, f))

for root, _, files in os.walk(positive_validation_directory):
    for f in files:
        if f.endswith("wav"):
            positive_validation_files.append(os.path.join(positive_validation_directory, f))

for root, _, files in os.walk(negative_training_directory):
    for f in files:
        if f.endswith("wav"):
            negative_training_files.append(os.path.join(negative_training_directory, f))

for root, _, files in os.walk(negative_validation_directory):
    for f in files:
        if f.endswith("wav"):
            negative_validation_files.append(os.path.join(negative_validation_directory, f))

# Try to do some MFCC stuff...
def write_ceps(ceps, fn):
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".ceps"
    np.save(data_fn, ceps)
    print("Written %s" % data_fn)

def create_ceps(fn):
    sample_rate, X = scipy.io.wavfile.read(fn)
    ceps, mspec, spec = mfcc(X)
    write_ceps(ceps, fn)

# Pass in a list of the positive and negative directoies here
def blind_read_ceps(base_dir):
    X = []
    file_names = []
    meow_dir = os.path.join(base_dir, "*.ceps.npy")
    print meow_dir
    file_list = glob.glob(meow_dir)
    file_number = 0
    for fn in file_list:
        print file_number
        file_number += 1
        ceps = np.load(fn)
        num_ceps = len(ceps)
        if not np.isnan(np.sum(ceps)):
            #print "Reading file:", fn
            X.append(np.mean(ceps[int(num_ceps * 1 / 10): int(num_ceps * 9 / 10)], axis = 0))
            file_names.append(fn)
        else:
            print "NaN detected in file:", fn
    print "Done with the files"
    return np.array(X), file_names


# Pass in a list of the positive and negative directoies here
def read_ceps(folder_list):
    X = []
    y = []
    for label, genre in enumerate(folder_list):
        meow_dir = os.path.join(genre, "*.ceps.npy")
        print meow_dir
        file_list = glob.glob(meow_dir)
        for fn in file_list:
            ceps = np.load(fn)
            num_ceps = len(ceps)
            if not np.isnan(np.sum(ceps)):
                X.append(np.mean(ceps[int(num_ceps * 1 / 10): int(num_ceps * 9 / 10)], axis = 0))
                y.append(label)
            else:
                print "NaN detected in file:", fn
    print np.array(X)
    print np.array(y)
    return np.array(X), np.array(y)

for f in positive_training_files:
    print(f)
    create_ceps(f)

for f in negative_training_files:
    print(f)
    create_ceps(f)

for f in positive_validation_files:
    print(f)
    create_ceps(f)

for f in negative_validation_files:
    print(f)
    create_ceps(f)


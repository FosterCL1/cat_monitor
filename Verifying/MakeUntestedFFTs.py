#! /usr/bin/python2.7
import os
import scipy.io.wavfile
import numpy as np
import glob
from matplotlib import pylab
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scikits.talkbox.features import mfcc

def create_fft(fn):
    sample_rate, X = scipy.io.wavfile.read(fn)
    fft_features = abs(scipy.fft(X)[:1000])
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".fft"
    np.save(data_fn, fft_features)

MEOW_DIR="/home/colin/Documents/Octave/CatMeow/"

def blind_read_fft(fn, base_dir):
    X = []
    y = []
    meow_dir = os.path.join(base_dir, "*.fft.npy")
    print meow_dir
    file_list = glob.glob(meow_dir)
    for fn in file_list:
        fft_features = np.load(fn)
        X.append(fft_features[:])
        y.append(label)
    return np.array(X), np.array(y)

def read_fft(fn, base_dir=MEOW_DIR, train_dir="Training"):
    X = []
    y = []
    for label, withMeow in enumerate(meow_list):
        print withMeow
        meow_dir = os.path.join(base_dir, withMeow, train_dir, "*.fft.npy")
        print meow_dir
        file_list = glob.glob(meow_dir)
        for fn in file_list:
            fft_features = np.load(fn)
            X.append(fft_features[:])
            y.append(label)
    return np.array(X), np.array(y)

unknown_file_directory = "/home/colin/Documents/Octave/CatMeow/Untested/"
unknown_file_files = []

for root, _, files in os.walk(unknown_file_directory):
    for f in files:
        if f.endswith("wav"):
            unknown_file_files.append(os.path.join(unknown_file_directory, f))

for f in unknown_file_files:
    print f
    create_fft(f)


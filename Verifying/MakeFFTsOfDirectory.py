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

meow_list = ["WithCat", "NoCat"]
MEOW_DIR="/home/colin/Documents/Octave/CatMeow/"

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

positive_training_directory = "WithCat/Training/"
positive_training_files = []
positive_validation_directory = "WithCat/Validation/"
positive_validation_files = []
negative_training_directory = "NoCat/Training/"
negative_training_files = []
negative_validation_directory = "NoCat/Validation/"
negative_validation_files = []
positive_all_directory = "WithCat2/AllFiles/"
positive_all_files = []
negative_all_directory = "NoCat2/AllFiles/"
negative_all_files = []
untested_all_directory = "Untested/"
untested_all_files = []

for root, _, files in os.walk(untested_all_directory):
    for f in files:
        if f.endswith("wav"):
            untested_all_files.append(os.path.join(untested_all_directory, f))

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


for root, _, files in os.walk(positive_all_directory):
    for f in files:
        if f.endswith("wav"):
            positive_all_files.append(os.path.join(positive_all_directory, f))

for root, _, files in os.walk(negative_all_directory):
    for f in files:
        if f.endswith("wav"):
            negative_all_files.append(os.path.join(negative_all_directory, f))

for f in positive_training_files:
    print f
    create_fft(f)

for f in negative_training_files:
    print f
    create_fft(f)

for f in positive_validation_files:
    print f
    create_fft(f)

for f in negative_validation_files:
    print f
    create_fft(f)

for f in positive_all_files:
    print f
    create_fft(f)

for f in negative_all_files:
    print f
    create_fft(f)

for f in untested_all_files:
    print f
    create_fft(f)

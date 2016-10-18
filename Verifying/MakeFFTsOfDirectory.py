#! /usr/bin/python2.7
import os
import scipy.io.wavfile
import numpy as np
import glob
import sys

def create_fft(fn):
    sample_rate, X = scipy.io.wavfile.read(fn)
    fft_features = abs(scipy.fft(X)[:1000])
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".fft"
    np.save(data_fn, fft_features)

if (len(sys.argv) < 1):
    print "You must enter at least one argument"

for arg in sys.argv[1:]:
    print "Using argument " + arg
    file_list = []
    for root, _, files in os.walk(arg):
        for f in files:
            if f.endswith("wav"):
                file_list.append(os.path.join(arg, f))

    for f in file_list:
        print f
        create_fft(f)


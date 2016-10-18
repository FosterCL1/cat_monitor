#! /usr/bin/python2.7
import os
import scipy.io.wavfile
import numpy as np
import glob
import sys
from matplotlib import pylab
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scikits.talkbox.features import mfcc

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
        print(f)
        create_ceps(f)


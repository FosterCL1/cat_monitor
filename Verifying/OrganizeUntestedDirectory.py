#! /usr/bin/python2.7
import os
import scipy.io.wavfile
import numpy as np
import glob
import pickle
import shutil
import ntpath
import sys
import getopt
import MakeMFCC
from matplotlib import pylab
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scikits.talkbox.features import mfcc
from sklearn import cross_validation

#meow_list = ["WithCat", "NoCat"]
#MEOW_DIR="/home/colin/Documents/Octave/CatMeow/"
UNTESTED_DIR="/home/colin/Documents/Octave/CatMeow/UntestedFiles/Untested/"
DEFAULT_TRAINING_FILE='objs.pickle'

def Usage():
    print "./OrganizeUntestedDirectory -i InputDirectory"

def blind_read_fft(base_dir):
    X = []
    file_names = []
    meow_dir = os.path.join(base_dir, "*.fft.npy")
    print meow_dir
    file_list = glob.glob(meow_dir)
    for fn in file_list:
        fft_features = np.load(fn)
        X.append(fft_features[:])
        file_names.append(fn)
    return np.array(X), file_names

#Plot some of the stuff...
def plot_confusion_matrix(cm, meow_list, name, title):
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(range(len(meow_list)))
    ax.set_xticklabels(meow_list)
    ax.xaxis.set_ticks_position("bottom")

def main(argv):
    input_directory = UNTESTED_DIR
    training_file = DEFAULT_TRAINING_FILE
    InputMethod = "FFT"

    try:
        opts, args = getopt.getopt(argv, "hmfi:", ["input="])
    except:
        print "Error gettiong options"
        Usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            Usage()
            sys.exit()
        elif opt in ('-m'):
            InputMethod="MEL"
        elif opt in ('-f'):
            InputMethod="FFT"
        elif opt in ('-i', '--input'):
            input_directory = arg

    with open('objs.pickle') as f:
        clf = pickle.load(f)

    if InputMethod == "FFT":
        print "Using FFT Training"
        X, file_names = blind_read_fft(input_directory)
        file_extension = 'fft.npy'
    elif InputMethod == "MEL":
        print "Using MFCC Training"
        X, file_names = MakeMFCC.blind_read_ceps(input_directory)
        file_extension = 'ceps.npy'
    else:
        print "No valid input method"
        sys.exit()

    y_pred = clf.predict(X)

    print y_pred

    pass_dir = "/home/colin/Documents/Octave/CatMeow/TestRun/WithCat/"
    fail_dir = "/home/colin/Documents/Octave/CatMeow/TestRun/NoCat/"
    # TODO: Verify output directories exist
    # TODO: Have these be inputs

    for fn, y in zip(file_names, y_pred):
        wav_file_name = ntpath.basename(fn.replace(file_extension, 'wav'))
        print "wav_file_name = " + wav_file_name
        original_wav_file = os.path.join(UNTESTED_DIR, wav_file_name)
        print "original_wav_file = " + original_wav_file
        if y == 0:
            out_fn = os.path.join(pass_dir, wav_file_name)
        else:
            out_fn = os.path.join(fail_dir, wav_file_name)
        print "Copying file " +  original_wav_file + " to " + out_fn
        shutil.copy2(original_wav_file, out_fn)
            
if __name__ == "__main__":
    main(sys.argv[1:])



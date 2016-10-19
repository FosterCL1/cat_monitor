#! /usr/bin/python2.7
import os
import scipy.io.wavfile
import numpy as np
import glob
import sys
import getopt
from matplotlib import pylab
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scikits.talkbox.features import mfcc
from sklearn import cross_validation
import MakeMFCC
import pickle

def read_training_ffts(fn, meow_list):
    X = []
    y = []
    for label, withMeow in enumerate(meow_list):
        meow_dir = os.path.join(withMeow, "*.fft.npy")
        print meow_dir
        file_list = glob.glob(meow_dir)
        for fn in file_list:
            fft_features = np.load(fn)
            X.append(fft_features[:])
            y.append(label)
    return np.array(X), np.array(y)

def Usage():
    print 'Usage: TraingAlgorithm.py -p positive_training_directory -n negative_training_directory -o output_file'


# Plot some of the stuff...
def plot_confusion_matrix(cm, meow_list, name, title):
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(range(len(meow_list)))

    ax.set_xticklabels(meow_list)
    ax.xaxis.set_ticks_position("bottom")

def main(argv):
    output_file = 'objs.pickle'
    InputMethod = "FFT"
    if argv < 3:
        Usage()
        sys.exit(2)

    try:
        opts, args = getopt.getopt(argv, "hmfp:n:o:", ["positive=","negative=","output="])
    except:
        Usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            Usage()
            sys.exit()
        elif opt in ('-p', '--positive'):
            positive_training_directory=arg
        elif opt in ('-n', '--negative'):
            negative_training_directory=arg
        elif opt in ('-o', '--output'):
            output_file=arg
        elif opt in ('-m'):
            InputMethod="MEL"
        elif opt in ('-f'):
            InputMethod="FFT"

    print "Using positive training directory ", positive_training_directory
    print "Using negative training directory ", negative_training_directory

    # Read the FFT data
    meow_list = [positive_training_directory, negative_training_directory]
    if InputMethod == "FFT":
        print "Using FFT Training"
        X, y = read_training_ffts(8000, meow_list)
    elif InputMethod == "MEL":
        print "Using MFCC Training"
        X, y = MakeMFCC.read_ceps(meow_list)
    else:
        print "No valid input method"
        sys.exit()

    # Split the data into test and train sets
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)

    # Perform the regression
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Test for an actual prediction
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    name = "TestName"
    title = "TestTitle"
    #plot_confusion_matrix(cm, meow_list, name, title)
    print "Confusion matrix of the data"
    print cm

    score = clf.score(X_test, y_test)
    print "Score"
    print score

    # If desired, this can be used to test alternate cross validation sets
    scores = cross_validation.cross_val_score(clf, X, y, cv=5)
    print scores

    print "Writing output to ", output_file
    with open(output_file, 'w') as f:
        pickle.dump(clf, f)

def lr_model(clf, X):
    return 1 / (1 + np.exp(-(clf.intercept_ + clf.coef_*X)))

#print("P(x=-1)=%.2f\tP(x=7)=%.2f" % (lr_model(clf, -1), lr_model(clf, 7)))

if __name__ == "__main__":
    main(sys.argv[1:])





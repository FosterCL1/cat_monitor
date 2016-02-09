#! /usr/bin/python2.7
import os
import scipy.io.wavfile
import numpy as np
import glob
from matplotlib import pylab
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scikits.talkbox.features import mfcc
from sklearn import cross_validation
import pickle

meow_list = ["WithCat2", "NoCat2"]
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

positive_all_directory = "WithCat2/AllFiles/"
negative_all_directory = "NoCat2/AllFiles/"

# Read everything
#X_train, y_train = read_fft(8000)
#X_test, y_test = read_fft(8000, train_dir="Validation")
X, y = read_fft(8000, train_dir="AllFiles")
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)

#Plot some of the stuff...
def plot_confusion_matrix(cm, meow_list, name, title):
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(range(len(meow_list)))
    ax.set_xticklabels(meow_list)
    ax.xaxis.set_ticks_position("bottom")

clf = LogisticRegression()
print clf

clf.fit(X_train,y_train)

#print(np.exp(clf.intercept_), np.exp(clf.coef_.ravel()))

def lr_model(clf, X):
    return 1 / (1 + np.exp(-(clf.intercept_ + clf.coef_*X)))

#print("P(x=-1)=%.2f\tP(x=7)=%.2f" % (lr_model(clf, -1), lr_model(clf, 7)))

y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
name = "TestName"
title = "TestTitle"
#plot_confusion_matrix(cm, meow_list, name, title)

print cm

score = clf.score(X_test, y_test)
print score

scores = cross_validation.cross_val_score(clf, X, y, cv=5)
print scores

with open('objs.pickle', 'w') as f:
    pickle.dump(clf, f)

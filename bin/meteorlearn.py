#!/usr/bin/env python3

# meteorlearn.py <wrongs dir> <rights dir>

import configparser, sklearn, os, sys
import numpy as np
import scipy.interpolate as interp
from glob import glob
from ast import literal_eval as make_tuple
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, RidgeClassifier, RidgeClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix

wrongs = [y for x in os.walk(sys.argv[1]) for y in glob(os.path.join(x[0], 'event.txt'))]
rights = [y for x in os.walk(sys.argv[2]) for y in glob(os.path.join(x[0], 'event.txt'))]

gnomonicx, gnomonicy, brightness, dct, target = [], [], [], [], []

for files in [wrongs, rights]:
    for o in files:
        obs = configparser.ConfigParser()
        obs.read(o)
        try:
            g = obs.get('trail', 'gnomonic').split()
            b = obs.get('trail', 'brightness').split()
            d = obs.get('trail', 'dct').split()
            assert(len(g) == len(b) and len(b) == len(d) and len(g) > 1)
            gnomonicx.append(map(lambda x:make_tuple(x)[0], g))
            gnomonicy.append(map(lambda x:make_tuple(x)[1], g))
            brightness.append(map(int, b))
            dct.append(map(int, d))
            target.append(files == rights)
        except:
            pass

np.random.seed(123456)
shuffle_index = np.random.permutation(len(target))
gnomonicx = np.array(gnomonicx)[shuffle_index]
gnomonicy = np.array(gnomonicy)[shuffle_index]
brightness = np.array(brightness)[shuffle_index]
dct = np.array(dct)[shuffle_index]
target = np.array(target)[shuffle_index]

def stretch_data(tab):
    max = np.amax([len(x) for x in tab])
    return [map(int, interp.interp1d(np.arange(len(x)), x)(np.linspace(0, len(x) - 1, max))) for x in tab]
    #return [np.pad(x, (0, (max - len(x))), 'wrap') for x in tab]

dct = stretch_data(dct)
gnomonicx = stretch_data(gnomonicx)
gnomonicy = stretch_data(gnomonicy)
brightness = stretch_data(brightness)
trainlen = len(target)*90/100
gnomonicx_train, gnomonicx_test = gnomonicx[:trainlen], gnomonicx[trainlen:]
gnomonicy_train, gnomonicy_test = gnomonicy[:trainlen], gnomonicy[trainlen:]
brightness_train, brightness_test = brightness[:trainlen], brightness[trainlen:]
dct_train, dct_test = dct[:trainlen], dct[trainlen:]
target_train, target_test = target[:trainlen], target[trainlen:]

dct_clf = GradientBoostingClassifier()
gnomonicx_clf = GradientBoostingClassifier()
gnomonicy_clf = GradientBoostingClassifier()
brightness_clf = GradientBoostingClassifier() #RandomForestClassifier()
dct_clf.fit(dct_train, target_train)
gnomonicx_clf.fit(gnomonicx_train, target_train)
gnomonicy_clf.fit(gnomonicy_train, target_train)
brightness_clf.fit(brightness_train, target_train)

print(cross_val_score(dct_clf, dct_train, target_train, cv=3, scoring="accuracy"))
print(cross_val_score(gnomonicx_clf, gnomonicx_train, target_train, cv=3, scoring="accuracy"))
print(cross_val_score(gnomonicy_clf, gnomonicy_train, target_train, cv=3, scoring="accuracy"))
print(cross_val_score(brightness_clf, brightness_train, target_train, cv=3, scoring="accuracy"))

#print(confusion_matrix(target_train, cross_val_predict(dct_clf, dct_train, target_train, cv=3)))
#print(confusion_matrix(target_train, cross_val_predict(gnomonicx_clf, gnomonicy_train, target_train, cv=3)))
#print(confusion_matrix(target_train, cross_val_predict(gnomonicy_clf, gnomonicy_train, target_train, cv=3)))
print(confusion_matrix(target_train, cross_val_predict(dct_clf, dct_train, target_train)))
print(confusion_matrix(target_test, cross_val_predict(dct_clf, dct_test, target_test)))

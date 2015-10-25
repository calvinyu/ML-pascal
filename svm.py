import numpy as np
import pylab as pl
import sys
from pylab import *
import matplotlib.pyplot as plt
from sklearn.utils import shuffle as sf
from sklearn.externals.six import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import ShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pydot
def read(filename):
	X = []
	Y = []
	f = open(filename, 'r')
	while True:
		chunk = f.readline()
		if chunk == "":
			break;
		chunk = chunk.split()
		y = chunk.pop(0)
		X.append(np.array(chunk))
		Y.append(y)
	return X,Y
def train(X, Y):
	clf = SVC()
	clf.fit(X,Y)
	return clf
if __name__ == "__main__":
	Xtrain, Ytrain = read('train')
	clf = train(Xtrain,Ytrain)
	Xtest, Ytest = read('test')
	Ypredict = clf.predict(Xtest)
	print np.array(Ytest)
	print np.array(Ypredict)

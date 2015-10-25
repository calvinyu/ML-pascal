from __future__ import division
import pickle
import numpy as np
import glob
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import pylab as pl
if __name__ == "__main__":
		dirTrue = glob.glob('../labels/*test*')
		trueLabels = []
		for dirName in dirTrue:
			trueLabels.append(pickle.load(open(dirName,'rb')))
		trueLabels = np.matrix(trueLabels)
		trueLabels = np.transpose(trueLabels)
		trueLabels = np.array(np.argmax(trueLabels,axis=1))
		trueLabels = np.concatenate(trueLabels, axis=1)

		dirNames = glob.glob('*_*')	
		for dirName in dirNames:
			fileName = glob.glob(dirName + "/*")
			#head = len(dirName) + 1
			#tail = 13 + len(dirName) + 1 + len(c)
			#tail = 19
			print dirName
			#print c
			#print fileName
			guess = []
			for f in fileName:
				guess.append(pickle.load(open(f,'rb')))
				#print (pickle.load(open(f,'rb')))
			#print len(guess)
			guess = np.concatenate(np.array(guess), axis=1)
			guess = np.transpose(guess)
			#print guess.shape
			maxGuess = np.argmax(guess, axis=0)
			#print maxGuess
			#for i in range(20):
			#	print i
			#	print np.sum(maxGuess==i)
			#print len(trueLabels)
			#print len(maxGuess)
			cm = confusion_matrix(trueLabels, maxGuess)
			cm = cm/np.amax(cm,axis=1)

			#print cm
			pl.matshow(cm)
			pl.title('Confusion matrix')
			pl.colorbar()
			pl.ylabel('True label')
			pl.xlabel('Predicted label')
			pl.show()

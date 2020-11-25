import pandas as pd
from sklearn import svm
from operator import itemgetter
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

def get_labels(datalist):
	labels = []
	documents = []
	for line in datalist:
		splittedline = line.split('\t')
		labels.append(splittedline[0])
		documents.append(splittedline[1].strip('\n'))

	return documents, labels
	
def get_gold(corpus_file):
	
	posglist, posdlist, postlist, negglist, negdlist, negtlist = [], [], [], [], [], []
	posg, posd, post, negg, negd, negt = 0, 0, 0, 0, 0, 0
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			if line.startswith("Positive"):
				if posg < 150:
					posglist.append(line.strip("\n"))
					posg += 1
				elif posd < 50:
					posdlist.append(line.strip("\n"))
					posd += 1
				elif post < 50:
					postlist.append(line.strip("\n"))
					post += 1
			elif line.startswith("Negative"):
				if negg < 150:
					negglist.append(line.strip("\n"))
					negg += 1
				elif negd < 50:
					negdlist.append(line.strip("\n"))
					negd += 1
				elif negt < 50:
					negtlist.append(line.strip("\n"))
					negt += 1
	goldlist = posglist + negglist
	devlist = posdlist + negdlist
	testlist = postlist + negtlist
	
	return goldlist, devlist, testlist
	
def read_unsup(corpus_file):
	unsupdocs = []
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			unsupdocs.append(line.strip('\n'))
					
	return unsupdocs
	
def identity(x):
    return x

def result_cap(results, threshmax, threshmin):
	
	tuplist, maxtup, mintup = [], [], []
	for index, result in enumerate(results):
		tuplist.append((result, index))
	for tup in tuplist:
		if tup[0] > threshmax:
			maxtup.append(tup)
			if len(maxtup) > 5:
				maxtup.sort(key=itemgetter(0), reverse=True)
				maxtup.pop()
		if tup[0] < threshmin:
			mintup.append(tup)
			if len(mintup) > 5:
				mintup.sort(key=itemgetter(0))
				mintup.pop()

	if len(maxtup) > len(mintup):
		maxtup.sort(key=itemgetter(0), reverse=True)
		mintup.sort(key=itemgetter(0))
		maxtup = maxtup[:len(mintup)]
	elif len(maxtup) < len(mintup):
		mintup.sort(key=itemgetter(0))
		maxtup.sort(key=itemgetter(0), reverse=True)
		mintup = mintup[:len(maxtup)]
	
	return maxtup, mintup


def main():
	
	p = True
	n = 0

	trainlist, devlist, testlist = get_gold("devnopreproc.txt")
	Xtrain, ytrain = get_labels(trainlist)
	Xdev, ydev = get_labels(devlist)
	Xtest, ytest = get_labels(testlist)
	Xunsup = read_unsup("nopunsupervised.txt")
	
	vec = TfidfVectorizer(preprocessor = identity, tokenizer = identity)
	classifier = Pipeline( [('vec', vec), ('cls', svm.LinearSVC())] )
	classifier.fit(Xtrain, ytrain)
	
	print("Starting length X:", len(Xtrain))
	threshmax, threshmin, updated = 1.25, -1.25, 0
	while p == True:
		Xtemp, ytemp = [], []
		samples = resample(Xunsup, replace=False, n_samples=10000, random_state=n)
		decision = classifier.decision_function(samples)
		maxtup, mintup = result_cap(decision, threshmax, threshmin)
		if maxtup != []:
			if len(Xtrain)+(len(maxtup)*2) > 400:
				diff = (len(Xtrain)+(len(maxtup)*2) - 400) / 2
				listlen = len(maxtup) - diff
				maxtup = maxtup[:int(listlen)]
				mintup = mintup[:int(listlen)]
			if len(Xtrain)+(len(maxtup)*2) <= 400:
				for item in maxtup:
					Xtemp.append(samples[item[1]].strip("\n"))
					ytemp.append("Positive")
				for item in mintup:
					Xtemp.append(samples[item[1]].strip("\n"))
					ytemp.append("Negative")
			
				threshmax += 0.002
				threshmin += -0.002
				n += 1

				vec2 = TfidfVectorizer(preprocessor = identity, tokenizer = identity)
				dummy = Pipeline( [('vec2', vec2), ('cls2', svm.LinearSVC())] )
				dummy.fit(Xtrain+Xtemp, ytrain+ytemp)
				Dguess = dummy.predict(Xdev)
				Yguess = classifier.predict(Xdev)
				acc, Dacc = accuracy_score(ydev, Yguess), accuracy_score(ydev, Dguess)
				if Dacc >= acc:
					Xtrain, ytrain = Xtrain+Xtemp, ytrain+ytemp
					print(len(Xtrain), Dacc)
					classifier.fit(Xtrain, ytrain)
					updated += 1
					if len(Xtrain) == 400:
						p = False
						print("Maximum size of 400 for X was reached. Current length X, thresholds, iteration and amount of times classifier updated:", len(Xtrain), threshmax, threshmin, n, updated)
				else:
					continue
			
		elif maxtup == []:
			print("Samples stopped exceeding threshold. Current size X, thresholds, iteration and amount of times classifier updated:", len(Xtrain), threshmax, threshmin, n, updated)
			p = False
	
	Ytestguess = classifier.predict(Xtest)
	print(classification_report(ytest, Ytestguess, digits=3))
	y_actu = pd.Series(ytest, name='Actual')
	y_pred = pd.Series(Ytestguess, name='Predicted')
	print(pd.crosstab(y_actu, y_pred))
	print("Accuracy:", accuracy_score(ytest, Ytestguess))
	
	""" For creating gold+bootstrap set
	thirtymil = read_unsup("nopsemistart.txt")
	tmlabels = classifier.predict(thirtymil)
	writefile = open("gbtaggedset.txt", "a")
	for index, tweet in enumerate(thirtymil):
		writefile.write(tmlabels[index]+'\t'+tweet+'\n')
	writefile.close()
	"""
	
main()

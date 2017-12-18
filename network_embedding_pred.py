from gensim.models.word2vec import Word2Vec
from sklearn import linear_model
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
import random


data = 'cdm/CDM'
model_name = 'nv'
for i in range(5):
	i = str(i)
	model = Word2Vec.load_word2vec_format('data/'+ data + '_' + model_name + '_' +i+'.embeddings', binary=False)
	X_train = []
	Y_train = []
	X_test = []
	Y_test = []
	X_test_pos = []
	X_test_neg = []
	with open ('data/'+ data + '_train_' +i+'.net','rb') as f:
		for line in f:
			pair = line.strip().split()
			if pair[0] not in model.vocab or pair[1] not in model.vocab:
				continue
			else:
				head = model[pair[0]]
				tail = model[pair[1]]
				feature = np.multiply(head,tail)
				X_train.append(feature)
				Y_train.append(1)

	with open ('data/' + data + '_train_neg_' + i + '.net','rb') as f:
		for line in f:
			pair = line.strip().split()
			if pair[0] not in model.vocab or pair[1] not in model.vocab:
				continue
			else:
				head = model[pair[0]]
				tail = model[pair[1]]
				feature = np.multiply(head,tail)
				X_train.append(feature)
				Y_train.append(-1)

	clf = linear_model.SGDClassifier(loss='squared_hinge')

	X_train = np.array(X_train)
	Y_train = np.array(Y_train)

	clf.fit(X_train,Y_train)

	Y_unknown = []
	Y_score_unknown = []
	with open ('data/'+ data + '_test_' + i + '.net','rb') as f:
		for line in f:
			pair = line.strip().split()
			if pair[0] not in model.vocab or pair[1] not in model.vocab:
				Y_unknown.append(1)
				Y_score_unknown.append(0.5)
			else:
				head = model[pair[0]]
				tail = model[pair[1]]
				feature = np.multiply(head,tail)
				X_test.append(feature)
				X_test_pos.append(feature)
				Y_test.append(1)

	with open ('data/'+ data + '_test_neg_' + i + '.net','rb') as f:
		for line in f:
			pair = line.strip().split()
			if pair[0] not in model.vocab or pair[1] not in model.vocab:
				Y_unknown.append(-1)
				Y_score_unknown.append(0.5)
			else:
				head = model[pair[0]]
				tail = model[pair[1]]
				feature = np.multiply(head,tail)
				X_test.append(feature)
				X_test_neg.append(feature)
				Y_test.append(-1)

	X_test = np.array(X_test)
	Y_test = np.array(Y_test)
	Y_pred = clf.decision_function(X_test)
	Y_pred_pos = clf.decision_function(X_test_pos)
	Y_pred_neg = clf.decision_function(X_test_neg)
	for i in range(len(Y_pred)):
		Y_pred[i] = 1 / (1 + np.exp(-Y_pred[i]))
	print roc_auc_score(Y_test, Y_pred)

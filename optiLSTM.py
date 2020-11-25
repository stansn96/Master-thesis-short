from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import keras
from keras import optimizers
from keras import constraints
from keras import initializers
from keras import regularizers
from keras import backend as K
from gensim.models import Word2Vec
from keras.regularizers import Regularizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.engine.topology import Layer, InputSpec
from keras.models import Sequential, Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.layers.convolutional import MaxPooling1D, Conv1D
from keras.layers import Dense, Dropout, LSTM, MaxPool1D, Flatten, Input, Bidirectional, TimeDistributed, Embedding, GlobalMaxPooling1D, GRU

import sys
import pickle
import numpy as np
from collections import Counter

class AttentionWithContext(Layer):

    def __init__(self, init='glorot_uniform', kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None, bias_constraint=None,  **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get('glorot_uniform')

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight((input_shape[-1], 1),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.b = self.add_weight((input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)

        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.built = True

    def compute_mask(self, input, mask):
        return None

    def call(self, x, mask=None):
        multData =  K.dot(x, self.kernel) # (x, 40, 1)
        multData = K.squeeze(multData, -1) # (x, 40)
        multData = multData + self.b # (x, 40) + (40,)

        multData = K.tanh(multData) # (x, 40)

        multData = multData * self.u # (x, 40) * (40, 1) => (x, 1)
        multData = K.exp(multData) # (X, 1)

        if mask is not None:
            mask = K.cast(mask, K.floatx()) #(x, 40)
            multData = mask*multData #(x, 40) * (x, 40, )

        multData /= K.cast(K.sum(multData, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        multData = K.expand_dims(multData)
        weighted_input = x * multData
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1],)

class AttLayer(Layer):
	def __init__(self, **kwargs):
		self.init = initializers.get('normal')
		super(AttLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		assert len(input_shape)==3
		self.W = self.init((input_shape[-1],))
		self.trainable_weights = [self.W]
		super(AttLayer, self).build(input_shape)

	def call(self, x, mask=None):
		eij = K.tanh(K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1))
		
		ai = K.exp(eij)
		weights = ai/K.expand_dims(K.sum(ai, axis=1),1)
        
		weighted_input = x*K.expand_dims(weights,2)
		return K.sum(weighted_input, axis=1)

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[-1])

def RNNClassifier(X_train, y_train, X_test, y_test):
	
	X = X_train+X_test
	c = Counter(word for x in X_train for word in x.split())
	X_train = [' '.join(y for y in x.split() if c[y] > 1) for x in X_train]
	
	y_train_reshaped = [1 if tmp_y=='Positive' else 0 for tmp_y in y_train]
	y_train_reshaped = to_categorical(np.asarray(y_train_reshaped))
	
	t = Tokenizer()
	t.fit_on_texts(X_train)
	vocab_size = len(t.word_index) + 1
	X_train = t.texts_to_sequences(X_train)
	max_length = max([len(s.split()) for s in X])
	X_train_reshaped = pad_sequences(X_train, maxlen=max_length, padding='post')
	
	embeddings_index = dict()
	f = open("unsup_embedding2.txt")
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	print('Loaded %s word vectors.' % len(embeddings_index))
	embedding_matrix = np.zeros((vocab_size, 200))
	for word, i in t.word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	
	embedding_layer = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=max_length, trainable=False, mask_zero = True)
	sequence_input = Input(shape=(max_length,), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)
	l_lstm = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
	l_drop = Dropout(0.75)(l_lstm)
	l_att = AttentionWithContext()(l_drop)
	preds = Dense(2, activation='linear')(l_att)
	model = Model(sequence_input, preds)
	model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.1), metrics=['acc'])
	print(model.summary())
	
	y_test_reshaped = [1 if tmp_y=='Positive' else 0 for tmp_y in y_test]
	y_test_reshaped = to_categorical(np.asarray(y_test_reshaped))
	X_test = t.texts_to_sequences(X_test)
	X_test_reshaped = pad_sequences(X_test, maxlen=max_length, padding='post')
	
	filepath = "best_gtset_model.h5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	
	model.fit(X_train_reshaped, y_train_reshaped, epochs=30, batch_size=16, validation_data=(X_test_reshaped, y_test_reshaped), callbacks=callbacks_list)
	loss, accuracy = model.evaluate(X_test_reshaped, y_test_reshaped, verbose=0)

	return loss, accuracy

def identity(x):
	return x

def get_datas(corpus_file):
	
	labels = []
	documents = []
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
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
	
	return testlist

def get_labels(datalist):
	labels = []
	documents = []
	for line in datalist:
		splittedline = line.split('\t')
		labels.append(splittedline[0])
		documents.append(splittedline[1].strip('\n'))

	return documents, labels

def main():

	testlist = get_gold("devnopreproc.txt")
	X_test, y_test = get_labels(testlist)
	#X_test, y_test = get_datas("devnopreproc.txt")
	X_train, y_train = get_datas("resizedgtagged.txt")
	loss, accuracy = RNNClassifier(X_train, y_train, X_test, y_test)
	print(loss, accuracy)

if __name__ == "__main__":
	main()	

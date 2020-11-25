import keras
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, LSTM, MaxPool1D, Flatten, Input, Bidirectional, TimeDistributed, Embedding, GlobalMaxPooling1D, GRU, Merge
from keras.layers.convolutional import MaxPooling1D, Conv1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import optimizers
from keras.regularizers import Regularizer
from keras.utils.np_utils import to_categorical
import numpy as np
import sys
import xml.etree.ElementTree as ET
from collections import Counter
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import pickle
from sklearn.model_selection import train_test_split

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

        multData =  K.dot(x, self.kernel)
        multData = K.squeeze(multData, -1)
        multData = multData + self.b

        multData = K.tanh(multData)

        multData = multData * self.u
        multData = K.exp(multData)

        if mask is not None:
            mask = K.cast(mask, K.floatx())
            multData = mask*multData

        multData /= K.cast(K.sum(multData, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        multData = K.expand_dims(multData)
        weighted_input = x * multData
        return K.sum(weighted_input, axis=1)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1],)

def get_datas(corpus_file):
	
	labels = []
	documents = []
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			splittedline = line.split('\t')
			labels.append(splittedline[0])
			documents.append(splittedline[1].strip('\n'))

	return documents, labels

def main():

	X_train, y_train = get_datas("resizedgtagged.txt")
	X_test, y_test = get_datas("testnopreproc.txt")

	X = X_train+X_test
	c = Counter(word for x in X_train for word in x.split())
	X_train = [' '.join(y for y in x.split() if c[y] > 1) for x in X_train]
	
	y_train_reshaped = [1 if tmp_y=='Positive' else 0 for tmp_y in y_train]
	y_train_reshaped = to_categorical(np.asarray(y_train_reshaped))
	
	t = Tokenizer()
	t.fit_on_texts(X_train)
	vocab_size = len(t.word_index) + 1
	X_train = t.texts_to_sequences(X_train)
	max_length = max([len(s.split()) for s in X]) - 1
	X_train_reshaped = pad_sequences(X_train, maxlen=max_length, padding='post')

	model = load_model("best_gtset_model.h5", custom_objects={'AttentionWithContext':AttentionWithContext})
	y_test_reshaped = [1 if tmp_y=='Positive' else 0 for tmp_y in y_test]
	y_test_reshaped = to_categorical(np.asarray(y_test_reshaped))
	X_test = t.texts_to_sequences(X_test)
	X_test_reshaped = pad_sequences(X_test, maxlen=max_length+1, padding='post')
	loss, accuracy = model.evaluate(X_test_reshaped, y_test_reshaped, verbose=10)
	print(loss,accuracy)

	
if __name__ == "__main__":
	main()		

from __future__ import print_function

import numpy
from keras.engine import Model
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Input
from keras import regularizers
from keras.layers.core import (Flatten, Dense, Lambda)
from keras.layers.wrappers import Bidirectional
import keras.backend as K

print('================  Prepare data...  ================')
print('')

word2sgmt = {}
seq = []
morphs = []
with open('train.data') as f:
    for line in f:
        word, sgmnts = line.split(':')
        sgmt = sgmnts.split('+')
        sgmt = list(s.split('-') for s in sgmt)
        word2sgmt[word] = sgmt
        seq.extend(sgmt)

for sgmt in seq:
    for morph in sgmt:
        morphs.append(morph)

print('number of words: ', len(word2sgmt))

morph_indices = dict((c, i) for i, c in enumerate(set(morphs)))
indices_morph = dict((i, c) for i, c in enumerate(set(morphs)))

print('number of morphemes: ', len(morphs))

x_train_1 = numpy.array([[1,2,3],[1,2,3]],numpy.int32)
x_train_2 = numpy.array([[2,4,1],[2,4,1]],numpy.int32)

print('================  Load pre-trained word vectors...  ================')
print('')

y_train = numpy.array([[4,5,2],[2,4,1]],numpy.int32)

print('================  Build model...  ================')
print('')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy
from scipy.spatial.distance import cosine

from keras.engine import Model
from keras.layers.embeddings import Embedding
from keras.layers import merge
from keras.layers.recurrent import LSTM
from keras.layers.core import Flatten, Dense, Activation, Lambda, Reshape
from keras.layers import Input
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras import regularizers
from keras.layers.merge import concatenate
import keras.backend as K

morph_seq_1 = Input(shape=(None,), dtype='int32', name='morph_seq_1')
morph_seq_2 = Input(shape=(None,), dtype='int32', name='morph_seq_2')

morph_embedding = Embedding(input_dim=10, output_dim=64, mask_zero=True)

embed_seq_1 = morph_embedding(morph_seq_1)
embed_seq_2 = morph_embedding(morph_seq_2)

biLSTM = Bidirectional(LSTM(3, dropout=0.2, recurrent_dropout=0.2, return_sequences=False), merge_mode='concat')

encoded_seq_1 = biLSTM(embed_seq_1)
encoded_seq_2 = biLSTM(embed_seq_2)

concat_vector = concatenate([encoded_seq_1, encoded_seq_2], axis=-1)
merge_vector = Reshape((2,6))(concat_vector)

seq_output = TimeDistributed(Dense(3))(merge_vector)

attention_1 = TimeDistributed(Dense(output_dim=3, activation='tanh', bias=False))(seq_output)

attention_2 = TimeDistributed(Dense(output_dim=1,
                                            activity_regularizer=regularizers.l1(0.01),
                                            bias=False))(attention_1)

def attn_merge(inputs, mask):
    vectors = inputs[0]
    logits = inputs[1]
    # Flatten the logits and take a softmax
    logits = K.squeeze(logits, axis=2)
    pre_softmax = K.switch(1, logits, -numpy.inf)
    weights = K.expand_dims(K.softmax(pre_softmax))
    return K.sum(vectors * weights, axis=1)


def attn_merge_shape(input_shapes):
    return (input_shapes[0][0], input_shapes[0][2])


attn = Lambda(attn_merge, output_shape=attn_merge_shape)
attn.supports_masking = True
attn.compute_mask = lambda inputs, mask: None
content_flat = attn([seq_output, attention_2])

model = Model(inputs=[morph_seq_1, morph_seq_2], outputs=content_flat)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit([x_train_1,x_train_2], y_train, 1)
from __future__ import print_function
import sys
import numpy
from keras.engine import Model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Flatten, Dense, Activation, Lambda, Reshape
from keras.layers import Input
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras import regularizers
from keras.layers.merge import concatenate
import keras.backend as K
from gensim.models import Word2Vec, KeyedVectors
from keras.utils import plot_model
from keras.preprocessing import sequence

number_of_segmentation = 2

gensim_model = "C:\\Users\\ahmetu\\Desktop\\Morphology Projects\\tvec.bin"

load_pretrained_vector = False

print('===================================  Prepare data...  ==============================================')
print('')

word2sgmt = {}
seq = []
morphs = []
with open('train.data') as f:
    for line in f:
        line = line.rstrip('\n')
        word, sgmnts = line.split(':')
        sgmt = sgmnts.split('+')
        sgmt = list(s.split('-') for s in sgmt)
        word2sgmt[word] = sgmt
        seq.extend(sgmt)

timesteps_max_len = 0

for sgmt in seq:
    if len(sgmt) > timesteps_max_len : timesteps_max_len = len(sgmt)
    for morph in sgmt:
        morphs.append(morph)

print ('number of words: ', len(word2sgmt))

morph_indices = dict((c, i+1) for i, c in enumerate(set(morphs)))
indices_morph = dict((i+1, c) for i, c in enumerate(set(morphs)))

print('number of morphemes: ', len(morphs))
print('number of unique morphemes: ', len(set(morphs)))

x_train = [[] for i in range(number_of_segmentation)]
for word in word2sgmt:
    for segmnt in word2sgmt[word]:
        x_train[word2sgmt[word].index(segmnt)].append([morph_indices[c] for c in segmnt])

for i in range(number_of_segmentation):
    x_train[i] = numpy.array(x_train[i])

print('shape of Xs before padding: ',x_train[0].shape, x_train[1].shape)

for i in range(len(x_train)):
    x_train[i] = sequence.pad_sequences(x_train[i], maxlen=timesteps_max_len)

print('shape of Xs after padding: ',x_train[0].shape, x_train[1].shape)

print('')
print('==========================  Load pre-trained word vectors...  ======================================')
print('')

# w2v_model = Word2Vec.load(gensim_model)
y_train = []

if load_pretrained_vector:
    w2v_model = KeyedVectors.load_word2vec_format(gensim_model, binary=True)
    for word in word2sgmt:
        y_train.append(w2v_model[word].tolist())
    y_train = numpy.array(y_train)
    if len(y_train) is not len(word2sgmt): sys.exit(
        'ERROR: Pre-trained vectors do not contain all words in wordlist !!')
    print('number of pre-trained vectors: ', len(w2v_model.vocab))
else:
    y_train = numpy.array([[4,5,2],[2,4,1],[2,4,1],[2,4,1],[4,6,2],[5,1,2]])

print('number of words found: ', len(y_train))

print('shapes of Y: ',y_train.shape)


print('')
print('===================================  Build model...  ===============================================')
print('')


morph_seq_1 = Input(shape=(None,), dtype='int32', name='morph_seq_1')
morph_seq_2 = Input(shape=(None,), dtype='int32', name='morph_seq_2')

morph_embedding = Embedding(input_dim=len(morphs), output_dim=3, mask_zero=True)

embed_seq_1 = morph_embedding(morph_seq_1)
embed_seq_2 = morph_embedding(morph_seq_2)

biLSTM = Bidirectional(LSTM(3, dropout=0.2, recurrent_dropout=0.2, return_sequences=False), merge_mode='concat')

encoded_seq_1 = biLSTM(embed_seq_1)
encoded_seq_2 = biLSTM(embed_seq_2)

concat_vector = concatenate([encoded_seq_1, encoded_seq_2], axis=-1)
merge_vector = Reshape((2,6))(concat_vector)

seq_output = TimeDistributed(Dense(3))(merge_vector)

attention_1 = TimeDistributed(Dense(units=3, activation='tanh', use_bias=False))(seq_output)

attention_2 = TimeDistributed(Dense(units=1,
                                            activity_regularizer=regularizers.l1(0.01),
                                    use_bias=False))(attention_1)

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
plot_model(model, show_shapes=True, to_file='model.png')

model.fit([x_train[i] for i in range(number_of_segmentation)], y_train, 1)

f_attn = K.function([model.layers[0].input, model.layers[1].input, K.learning_phase()], [model.layers[-2].output])

attention_weights = f_attn([x_train[0],x_train[1],0])[0]

print('attention weights without softmax:\n', attention_weights)

'''
print('')
print('====================================================================================================')
print('')

f_lsmt_1 = K.function([model.layers[0].input, model.layers[1].input, K.learning_phase()], [model.layers[3].get_output_at(0)])
f_lsmt_2 = K.function([model.layers[0].input, model.layers[1].input, K.learning_phase()], [model.layers[3].get_output_at(1)])

lstm_weights_1 = f_lsmt_1([x_train[0],x_train[1],0])[0]
lstm_weights_2 = f_lsmt_2([x_train[0],x_train[1],0])[0]

print('lstm 1 weights:\n', lstm_weights_1)
print('')

print('lstm 2 weights:\n', lstm_weights_2)
print('')

f_conc = K.function([model.layers[0].input, model.layers[1].input, K.learning_phase()], [model.layers[4].output])

conc_weights = f_conc([x_train[0],x_train[1],0])[0]

print('concatenated weights:\n', conc_weights)
print('')
f_reshape = K.function([model.layers[0].input, model.layers[1].input, K.learning_phase()], [model.layers[5].output])

reshape_weights = f_reshape([x_train[0],x_train[1],0])[0]

print('concatenated weights:\n', reshape_weights)
'''
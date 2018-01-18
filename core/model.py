from __future__ import print_function
from __future__ import unicode_literals

import codecs
import sys

import keras.backend as K
import numpy
from gensim.models import KeyedVectors
from keras import regularizers
from keras.engine import Model
from keras.layers import Input
from keras.layers.core import Dense, Lambda, Reshape, Masking
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.preprocessing import sequence
from keras.utils import plot_model
import resource

number_of_segmentation = 10

vector = sys.argv[1]

gensim_model = vector

load_pretrained_vector = True

print('===================================  Prepare data...  ==============================================')
print('')

word2sgmt = {}
word2segmentations = {}
seq = []
morphs = []

f = codecs.open(sys.argv[0], encoding='utf-8')
for line in f:
    line = line.rstrip('\n')
    word, sgmnts = line.split(':')
    sgmt = sgmnts.split('+')
    word2segmentations[word] = list(s for s in sgmt)
    sgmt = list(s.split('-') for s in sgmt)
    word2sgmt[word] = sgmt
    seq.extend(sgmt)

timesteps_max_len = 0

for sgmt in seq:
    if len(sgmt) > timesteps_max_len: timesteps_max_len = len(sgmt)
    for morph in sgmt:
        morphs.append(morph)

print('number of words: ', len(word2sgmt))

morph_indices = dict((c, i + 1) for i, c in enumerate(set(morphs)))
morph_indices['###'] = 0

indices_morph = dict((i+1, c) for i, c in enumerate(set(morphs)))

print('number of morphemes: ', len(morphs))
print('number of unique morphemes: ', len(set(morphs)))

x_train = [[] for i in range(number_of_segmentation)]
for word in word2sgmt:
    for i in range(len(word2sgmt[word])):
        x_train[i].append([morph_indices[c] for c in word2sgmt[word][i]])

for i in range(number_of_segmentation):
    x_train[i] = numpy.array(x_train[i])

for i in range(len(x_train)):
    x_train[i] = sequence.pad_sequences(x_train[i], maxlen=timesteps_max_len)

print('')
print('==========================  Load pre-trained word vectors...  ======================================')
print('')

# w2v_model = Word2Vec.load(gensim_model)
y_train = []

if load_pretrained_vector:
    w2v_model = KeyedVectors.load_word2vec_format(gensim_model, binary=False, encoding='utf-8')
    for word in word2sgmt:
        y_train.append(w2v_model[word].tolist())
    y_train = numpy.array(y_train)
    if len(y_train) != len(word2sgmt): sys.exit(
        'ERROR: Pre-trained vectors do not contain all words in wordlist !!')
    print('number of pre-trained vectors: ', len(w2v_model.vocab))
else:
    y_train = numpy.array([[4, 5, 2], [2, 4, 1], [2, 4, 1], [2, 4, 1], [4, 6, 2], [5, 1, 2]])

print('number of words found: ', len(y_train))

print('shape of Y: ', y_train.shape)

print('')
print('===================================  Save Input and Output...  ===============================================')
print('')

numpy.save("x_train", x_train)
numpy.save("y_train", y_train)

print('')
print('===================================  Build model...  ===============================================')
print('')

morph_seg = []
for i in range(number_of_segmentation):
    morph_seg.append(Input(shape=(None,), dtype='int32'))

morph_embedding = Embedding(input_dim=len(set(morphs))+1, output_dim=50, mask_zero=True, name="embeddding")

embed_seg = []
for i in range(number_of_segmentation):
    embed_seg.append(morph_embedding(morph_seg[i]))

biLSTM = Bidirectional(LSTM(200, dropout=0.2, recurrent_dropout=0.2, return_sequences=False), merge_mode='concat')

encoded_seg = []
for i in range(number_of_segmentation):
    encoded_seg.append(biLSTM(embed_seg[i]))

concat_vector = concatenate(encoded_seg, axis=-1)
merge_vector = Reshape((number_of_segmentation, 400))(concat_vector)

masked_vector = Masking()(merge_vector)

seq_output = TimeDistributed(Dense(200))(masked_vector)

attention_1 = TimeDistributed(Dense(units=200, activation='tanh', use_bias=False))(seq_output)

attention_2 = TimeDistributed(Dense(units=1,
                                    activity_regularizer=regularizers.l1(0.01),
                                    use_bias=False))(attention_1)


def attn_merge(inputs, mask):
    vectors = inputs[0]
    logits = inputs[1]
    # Flatten the logits and take a softmax
    logits = K.squeeze(logits, axis=2)
    pre_softmax = K.switch(mask[0], logits, -numpy.inf)
    weights = K.expand_dims(K.softmax(pre_softmax))
    return K.sum(vectors * weights, axis=1)


def attn_merge_shape(input_shapes):
    return (input_shapes[0][0], input_shapes[0][2])


attn = Lambda(attn_merge, output_shape=attn_merge_shape)
attn.supports_masking = True
attn.compute_mask = lambda inputs, mask: None
content_flat = attn([seq_output, attention_2])

model = Model(inputs=morph_seg, outputs=content_flat)

model.compile(loss='cosine_proximity', optimizer='adam', metrics=['accuracy'])

model.summary()
# plot_model(model, show_shapes=True, to_file='model.png')

model.fit(x=x_train, y=y_train, batch_size=int(sys.argv[2]), epochs=int(sys.argv[3]))

print('')
print('===================================  Save model weights...  ===============================================')
print('')

model.save_weights("weights.h5")

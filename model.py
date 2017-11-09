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
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.preprocessing import sequence
from keras.utils import plot_model
import resource


'''
 def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    print(get_memory() * 1024 / 20)
    resource.setrlimit(resource.RLIMIT_AS, (10000000000, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

memory_limit()
'''
number_of_segmentation = 10

vector_tr = sys.argv[1]
vector_eng = '/Users/ahmet/Desktop/Corpus/GoogleNews-vectors-negative300.bin'

gensim_model = vector_tr

load_pretrained_vector = True

print('===================================  Prepare data...  ==============================================')
print('')

word2sgmt = {}
word2segmentations = {}
seq = []
morphs = []

f = codecs.open('input_96K_35_20.txt', encoding='utf-8')
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
    if len(sgmt) > timesteps_max_len : timesteps_max_len = len(sgmt)
    for morph in sgmt:
        morphs.append(morph)

print ('number of words: ', len(word2sgmt))

morph_indices = dict((c, i+1) for i, c in enumerate(set(morphs)))
indices_morph = dict((i+1, c) for i, c in enumerate(set(morphs)))

print('morpheme indices: ', indices_morph)

print('number of morphemes: ', len(morphs))
print('number of unique morphemes: ', len(set(morphs)))

x_train = [[] for i in range(number_of_segmentation)]
for word in word2sgmt:
    for i in range(len(word2sgmt[word])):
        x_train[i].append([morph_indices[c] for c in word2sgmt[word][i]])

print('')
for i in range(number_of_segmentation):
    print(i,'. Group Segmentations : ', x_train[i])
print('')

for i in range(number_of_segmentation):
    x_train[i] = numpy.array(x_train[i])

print('shape of Xs before padding')
for i in range(number_of_segmentation):
    print(x_train[i].shape)

for i in range(len(x_train)):
    x_train[i] = sequence.pad_sequences(x_train[i], maxlen=timesteps_max_len)

print('')
print('shape of Xs after padding')
for i in range(number_of_segmentation):
    print(x_train[i].shape)

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
    y_train = numpy.array([[4,5,2],[2,4,1],[2,4,1],[2,4,1],[4,6,2],[5,1,2]])

print('number of words found: ', len(y_train))

print('shape of Y: ', y_train.shape)


print('')
print('===================================  Build model...  ===============================================')
print('')

'''
morph_seq_1 = Input(shape=(None,), dtype='int32', name='morph_seq_1')
morph_seq_2 = Input(shape=(None,), dtype='int32', name='morph_seq_2')
'''

morph_seg = []
for i in range(number_of_segmentation):
    morph_seg.append(Input(shape=(None,), dtype='int32'))

morph_embedding = Embedding(input_dim=len(morphs), output_dim=50, mask_zero=True)

'''
embed_seq_1 = morph_embedding(morph_seq_1)
embed_seq_2 = morph_embedding(morph_seq_2)
'''

embed_seg = []
for i in range(number_of_segmentation):
    embed_seg.append(morph_embedding(morph_seg[i]))

biLSTM = Bidirectional(LSTM(200, dropout=0.2, recurrent_dropout=0.2, return_sequences=False), merge_mode='concat')

'''
encoded_seq_1 = biLSTM(embed_seq_1)
encoded_seq_2 = biLSTM(embed_seq_2)
'''

encoded_seg = []
for i in range(number_of_segmentation):
    encoded_seg.append(biLSTM(embed_seg[i]))

concat_vector = concatenate(encoded_seg, axis=-1)
merge_vector = Reshape((number_of_segmentation,400))(concat_vector)

seq_output = TimeDistributed(Dense(200))(merge_vector)

attention_1 = TimeDistributed(Dense(units=200, activation='tanh', use_bias=False))(seq_output)

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

model = Model(inputs=morph_seg, outputs=content_flat)

model.compile(loss='cosine_proximity', optimizer='adam', metrics=['accuracy'])

model.summary()
#plot_model(model, show_shapes=True, to_file='model.png')

model.fit(x=x_train, y=y_train, batch_size=int(sys.argv[2]), epochs=int(sys.argv[3]))

f_attn = K.function([model.layers[0].input, model.layers[1].input, model.layers[2].input, model.layers[3].input,
model.layers[4].input,model.layers[5].input,model.layers[6].input,model.layers[7].input,model.layers[8].input,model.layers[9].input,
                     K.learning_phase()],
                    [model.layers[-2].output])

attention_weights = f_attn([x_train[0],x_train[1],x_train[2],x_train[3],x_train[4],x_train[5],x_train[6],x_train[7],x_train[8],x_train[9],0])[0]

print('')
# print('attention weights without softmax:\n', attention_weights)
print('')


def attn_softmax(y, axis=None):
    y = y - numpy.expand_dims(numpy.max(y, axis=axis), axis)
    y = numpy.exp(y)
    ax_sum = numpy.expand_dims(numpy.sum(y, axis=axis), axis)
    return y / ax_sum

attention_soft_weights = attn_softmax(attention_weights,axis=1)

# print('attention weights with softmax:\n', attention_soft_weights)

print('')
print('====================================================================================================')
print('')

max = [0] * len(attention_soft_weights)
selecteds = [0] * len(attention_soft_weights)
for i in range(len(attention_soft_weights)):
    for j in range(len(attention_soft_weights[i])):
        if max[i] < attention_soft_weights[i][j]:
            max[i] = attention_soft_weights[i][j]
            selecteds[i] = j


indx = 0
segmentations = {}
file = codecs.open("selected_segmentations_" + sys.argv[2] + "_"+ sys.argv[3] +".txt", "w", "utf-8")

for word in word2sgmt:
    segmentations[word]= word2segmentations[word][selecteds[0]]
    #print(word +' : '+ segmentations[word])
    file.write(word +' : '+ segmentations[word] + "\n")
    indx = indx + 1

file.close()


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
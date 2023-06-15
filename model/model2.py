import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from tf2crf import CRF, ModelWithCRFLoss
from sklearn.preprocessing import MultiLabelBinarizer
from keras.callbacks import ModelCheckpoint
from pathlib import Path
import tensorflow_addons as tfa

import gc

# DATADIR is the path to the data directory
DATADIR = '../data'

#  hyperparameters used in the model
params = {
    'dim_chars': 100,
    'dim': 300,
    'dropout': 0.2106,
    'num_oov_buckets': 1,
    'epochs': 25,
    'batch_size': 20,
    'buffer': 15000,
    'filters': 30,
    'kernel_size': 3,
    'lstm_size': 100,
    'words': str(Path(DATADIR, 'vocab.words.txt')),
    'chars': str(Path(DATADIR, 'vocab.chars.txt')),
    'tags': str(Path(DATADIR, 'vocab.tags.txt')),
    'glove': str(Path(DATADIR, 'glove.npz'))
}

""" Data Loading """

# Load word vocabulary file and create dictionary mapping each word to an index
with open(params['words'], 'r') as f:
    word_vocab = [line.strip() for line in f]
word2idx = {word: idx + 2 for idx, word in enumerate(word_vocab)}
word2idx['PAD'] = 0
word2idx['UNK'] = params['num_oov_buckets']


# Load the training data and convert the sentences to a sequence of word and character
# indices using the dictionaries created above
def load_data(filename):
    with open(filename, 'r') as f:
        sentences = [line.replace('\t', ' ').strip().split() for line in f]
    return sentences

# Find the maximum length of a character in a word and the maximum length of a sentence
max_len_char = 50
max_len_sentence = 18

# Create the layer, passing the vocab directly. You can also pass the
# vocabulary arg a path to a file containing one vocabulary word per
# line.
vectorize_text_layer = tf.keras.layers.TextVectorization(
 output_mode='int',
 split='whitespace',
 standardize=None,
 vocabulary=params['words'])

vectorize_char_layer = tf.keras.layers.TextVectorization(
 output_mode='int',
 standardize=None,
 split='character',
 output_sequence_length=max_len_char,
 vocabulary=params['chars'])

vectorize_tag_layer = tf.keras.layers.TextVectorization(
 output_mode='int',
 standardize=None,
 vocabulary=params['tags'])


train_words = tf.data.TextLineDataset(os.path.join(DATADIR, 'train.words.txt'))
train_tags = tf.data.TextLineDataset(os.path.join(DATADIR, 'train.tags.txt'))
#train_dataset = tf.data.Dataset.zip((tf.data.TextLineDataset(os.path.join(DATADIR, 'train.words.txt')),tf.data.TextLineDataset(os.path.join(DATADIR, 'train.tags.txt'))))


def lbl(text, label):
    data = dict()
    data['word_in'] = tf.cast(text[0], dtype=tf.float32)
    data['char_in'] = tf.cast(text[1], dtype=tf.float32)
    return data, tf.cast(label,dtype=tf.float32)

def pad_after(t, max_in_dims, constant_values=0):
    diff = max_in_dims - tf.shape(t)
    paddings = tf.pad(diff[:, None], [[0, 0], [1, 0]])
    return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)
def pad_before(t, max_in_dims, constant_values=0):
    diff = max_in_dims - tf.shape(t)
    paddings = tf.pad(diff[:, None], [[0, 0], [0, 1]])
    return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)



#print(vectorize_char_layer("exceljs Project exceljs 0.1.0 for Node.js".split()))
train_chars = train_words.map(lambda x: (vectorize_char_layer(pad_after(tf.strings.split(x), max_len_sentence, 0))))
train_data = train_words.map(lambda x: pad_before((vectorize_text_layer(x)),max_len_sentence))
train_tags = train_tags.map(lambda x: pad_before(vectorize_tag_layer(x),max_len_sentence))
ds = tf.data.Dataset.zip(((train_data,train_chars), train_tags))


train_dataset = ds.repeat(3).shuffle(buffer_size=1024,reshuffle_each_iteration=False).padded_batch(params['batch_size'],drop_remainder=True).map(lbl)



AUTOTUNE = tf.data.AUTOTUNE

def configure_dataset(dataset):
  return dataset.cache().prefetch(buffer_size=AUTOTUNE)


train_dataset = configure_dataset(train_dataset)
#train_data = configure_dataset(train_data)
#train_tags = configure_dataset(train_tags)

#train_chars = train_chars.map(lambda x: x.astype('float32'))
#train_data = train_data.astype('float32')
#train_tags = train_tags .astype('float32')


# Load the pre-trained GloVe embeddings from the "glove.npz"
# file and create an embedding matrix using the word indices.
embeddings = np.load(params['glove'])['embeddings']
embedding_matrix = np.zeros((len(word_vocab) + 2, 300))
for word, i in word2idx.items():
    try:
        embedding_vector = embeddings[i]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    except:
        pass


"""
Model Building 
The model takes as input both words and characters, and uses an embedding layer for the words and a convolutional layer for the characters. The output of the model is passed through a bidirectional LSTM layer followed by a CRF layer, which allows for joint optimization of the sequence labeling task. Finally, a custom loss function is used to optimize the model during training.

The model architecture includes the following layers:

* An `embedding layer` for words.
* A `convolutional layer` for characters.
* A `bidirectional LSTM` layer.
* A` CRF layer` for joint optimization of the sequence labeling task.
"""


def create_model(params_):
    # Define input layers for words and characters
    word_in = tf.keras.layers.Input(shape=(params_["max_len_sentence"],), name='word_in')
    char_in = tf.keras.layers.Input(shape=(params_["max_len_sentence"], params_["max_len_char"],), name='char_in')

    # Create word embeddings layer and load pre-trained embeddings
    emb_word = tf.keras.layers.Embedding(input_dim=params_["len_word_vocab"], output_dim=params_['dim'], weights=[embedding_matrix], trainable=False)(word_in)

    # Create character embeddings layer and apply convolutional layers
    emb_char = tf.keras.layers.TimeDistributed(tf.keras.layers.Embedding(input_dim=params_["len_char_vocab"], output_dim=30))(char_in)
    char_enc = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=params_['filters'], kernel_size=params_['kernel_size']))(emb_char)
    char_flat = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(char_enc)

    # Concatenate word and character embeddings and pass through bidirectional LSTM
    x = tf.keras.layers.concatenate([emb_word, char_flat])
    main_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=params_['lstm_size'], return_sequences=True))(x)

    # Apply dropout and use CRF layer for final output
    main_lstm_dropout = tf.keras.layers.Dropout(rate=params_['dropout'])(main_lstm)
    crf = CRF(params_["len_tag_vocab"], dtype='float32')
    out = crf(main_lstm_dropout)

    # Create the base model and wrap it with ModelWithCRFLoss for training
    base_model = tf.keras.Model([word_in, char_in], out)
    return ModelWithCRFLoss(base_model, sparse_target=True)



print('creating model')
# Create the base model and wrap it with ModelWithCRFLoss for training
params["max_len_sentence"] = max_len_sentence
params["max_len_char"] = max_len_char
params["len_char_vocab"] = vectorize_char_layer.vocabulary_size()
params["len_word_vocab"] = vectorize_text_layer.vocabulary_size()
params["len_tag_vocab"] = vectorize_tag_layer.vocabulary_size()
model = create_model(params)

print(params["len_char_vocab"]) #66
print(params["len_word_vocab"]) #164653
print(params["len_tag_vocab"] ) #8

# Compile the model using 'adam' optimizer and 'accuracy' as the metric to evaluate the model performance
model.compile(optimizer='adam', metrics=['accuracy'])

# Define the EarlyStopping callback to monitor 'f1_score', stop when there is no improvement after 'patience' epochs
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='f1_score', mode='max', patience=3)
DATASET_SIZE = len(list(open(os.path.join(DATADIR, 'train.words.txt'))))
STEP_SIZE = DATASET_SIZE // params['batch_size']
print(DATASET_SIZE)

train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.3 * DATASET_SIZE)
training_dataset = train_dataset.take(train_size)
val_dataset = train_dataset.skip(train_size)

print('fitting model')
# Fit the model with the defined training data and validation split, along with the defined callbacks
history = model.fit(training_dataset,
                    epochs=3,
                    steps_per_epoch=STEP_SIZE,
                    validation_data= val_dataset,
                    callbacks=earlystopping)

test_words = tf.data.TextLineDataset(os.path.join(DATADIR, 'testa.words.txt'))
test_tags = tf.data.TextLineDataset(os.path.join(DATADIR, 'testa.tags.txt'))


test_chars = test_words.map(lambda x: (vectorize_char_layer(pad_after(tf.strings.split(x), max_len_sentence, 0))))
test_data = test_words.map(lambda x: pad_before((vectorize_text_layer(x)),max_len_sentence))
test_tags = test_tags.map(lambda x: pad_before(vectorize_tag_layer(x),max_len_sentence))
ds = tf.data.Dataset.zip(((test_data,test_chars),test_tags))
test_dataset = ds.padded_batch(params['batch_size']).map(lbl)

# Evaluate the model on the test data
print(model.evaluate(test_dataset))
#model.save('cpe_model_2')
y_pred = model.predict([test_data, test_chars])


class vectorize_padded_chars(tf.keras.layers.Layer):
    def __init__(self, text_vectorization_layer,max_len_word,max_len_char):
        super(vectorize_padded_chars, self).__init__()
        self.char_vectorization_layer = text_vectorization_layer
        self.lWord = max_len_word
        self.lChar = max_len_char

    def call(self, inputs):
        vectorized_input = self.char_vectorization_layer(tf.strings.split(inputs))
        diff = [self.lWord, self.lChar] - tf.shape(vectorized_input)
        out = (tf.pad(vectorized_input, tf.pad(diff[:, None], [[0, 0], [1, 0]]), 'CONSTANT', constant_values=0))
        return out


class vectorize_padded_words(tf.keras.layers.Layer):
    def __init__(self, text_vectorization_layer, max_len_word):
        super(vectorize_padded_words, self).__init__()
        self.text_vectorization_layer = text_vectorization_layer
        self.lWord = max_len_word

    def call(self, inputs):
        vectorized_input = self.text_vectorization_layer(inputs)
        diff = self.lWord - tf.shape(vectorized_input)
        out = (tf.pad(vectorized_input, tf.pad(diff[:, None], [[0, 0], [0, 1]]), 'CONSTANT', constant_values=0))
        return out


class contextualize(tf.keras.layers.Layer):
    def __init__(self, tag_vectorization_layer):
        super(contextualize, self).__init__()
        self.tag_vocab = tf.constant(tag_vectorization_layer.get_vocabulary())

    def lookup(self,tensor):
        return (tf.gather(self.tag_vocab, indices=tensor))

    def call(self, inputs):
        text = inputs[0]
        predictions = inputs[1]
        slice_tags = predictions[0][-(tf.shape(tf.strings.split(text))[0]):]
        tags_pred = tf.map_fn(self.lookup, slice_tags, dtype=tf.int32,fn_output_signature=tf.string)
        return tf.transpose([tf.strings.split(text), tags_pred])


lines = 'Adobe Media Encoder 15.4 for Windows' # this one works as it's familiar
# this one will fail: lines = "microsoftsunknown edgeupdater 1518.107"
# this one will fail: lines = "notaknownword edgeupdater 1518.107"

vec_char = vectorize_padded_chars(vectorize_char_layer, params["max_len_sentence"], params['max_len_char'])
vec_word = vectorize_padded_words(vectorize_text_layer, params["max_len_sentence"])
cont_layer = contextualize(tag_vectorization_layer=vectorize_tag_layer)

inputs = tf.keras.Input(type_spec=tf.TensorSpec(shape=(),dtype=tf.string),name="text")
X_test_char = tf.cast(vec_char(inputs), dtype=tf.float32)
X_test_word = tf.cast(vec_word(inputs), dtype=tf.float32)
#model = tf.keras.models.load_model('cpe_model_2')
outputs = model({'word_in':[X_test_word],'char_in':[X_test_char]})
outputs = cont_layer([inputs,outputs])
inference_model = tf.keras.Model([inputs], outputs)


#test a prediction
print(inference_model({'text': tf.constant(lines)}))

# save the final model

# https://www.tensorflow.org/api_docs/python/tf/keras/Model#save
inference_model.save(
  "cpe_model_3",
)

exit()


import numpy as np

from utils import get_char_vector_dict, encode_labels, vectorize_charseq

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.optimizers import Adagrad
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Convolution1D, MaxPooling1D

def vectorize(tokens, std_token_len,
              nb_left_tokens, left_char_len,
              nb_right_tokens, right_char_len,
              char_vector_dict = {},
              ):
    
    if not char_vector_dict:
        char_vector_dict = get_char_vector_dict(tokens)

    left_X, tokens_X, right_X = [], [], []

    for token_idx, token in enumerate(tokens):
        # ignore boundary markers:
        if token in ("@", "$"):
            continue

        # vectorize target token:
        tokens_X.append(vectorize_charseq(token, char_vector_dict, std_token_len))
        # vectorize left context:

        left_context = [tokens[token_idx-(t+1)] for t in range(nb_left_tokens)
                             if token_idx-(t+1) >= 0][::-1]
        left_str = " ".join(left_context)
        left_X.append(vectorize_charseq(left_str, char_vector_dict, left_char_len))

        # vectorize right context:
        right_str = " ".join([tokens[token_idx+t+1] for t in range(nb_right_tokens)
                             if token_idx+t+1 < len(tokens)])
        right_X.append(vectorize_charseq(right_str, char_vector_dict, right_char_len))
    
    tokens_X = np.asarray(tokens_X, dtype="int8")
    left_X = np.asarray(left_X, dtype="int8")
    right_X = np.asarray(right_X, dtype="int8")

    return left_X, tokens_X, right_X, char_vector_dict

def build_lemmatizer(nb_filters,
                    std_token_len,
                    filter_length,
                    char_vector_dict,
                    nb_lemmas,
                    nb_postags,
                    dense_dims):
    m = Graph()

    # specify inputs:
    m.add_input(name='left_input', ndim=3)
    m.add_input(name='token_input', ndim=3)
    m.add_input(name='right_input', ndim=3)

    # add left context nodes:
    m.add_node(Convolution1D(input_dim=len(char_vector_dict),
                            nb_filter=nb_filters,
                            filter_length=filter_length,
                            activation="relu",
                            border_mode="valid",
                            subsample_length=1), 
                   name="left_conv1", input='left_input')
    m.add_node(MaxPooling1D(pool_length=2),
                   name="left_pool1", input="left_conv1")
    m.add_node(LSTM(nb_filters/2, dense_dims),
                   name='left_lstm1', input='left_pool1')
    m.add_node(Dropout(0.5),
                   name='left_dropout1', input='left_lstm1')
    m.add_node(Activation('relu'),
                   name='left_relu1', input='left_dropout1')

    # add right context nodes:
    m.add_node(Convolution1D(input_dim=len(char_vector_dict),
                            nb_filter=nb_filters,
                            filter_length=filter_length,
                            activation="relu",
                            border_mode="valid",
                            subsample_length=1), 
                   name="right_conv1", input='right_input')
    m.add_node(MaxPooling1D(pool_length=2),
                   name="right_pool1", input="right_conv1")
    m.add_node(LSTM(nb_filters/2, dense_dims),
                   name='right_lstm1', input='right_pool1')
    m.add_node(Dropout(0.5),
                   name='right_dropout1', input='right_lstm1')
    m.add_node(Activation('relu'),
                   name='right_relu1', input='right_dropout1')

    # add target token nodes:
    m.add_node(Convolution1D(input_dim=len(char_vector_dict),
                            nb_filter=nb_filters,
                            filter_length=filter_length,
                            activation="relu",
                            border_mode="valid",
                            subsample_length=1), 
                   name="token_conv1", input='token_input')
    m.add_node(MaxPooling1D(pool_length=2),
                   name="token_pool1", input="token_conv1")
    m.add_node(Flatten(),
                   name="token_flatten1", input="token_pool1")
    m.add_node(Dropout(0.5),
                   name="token_dropout1", input="token_flatten1")
    output_size =  nb_filters * (((std_token_len-filter_length)/1)+1)/2
    m.add_node(Dense(output_size, dense_dims),
                   name="token_dense1", input="token_dropout1")
    m.add_node(Dropout(0.5),
                   name="token_dropout2", input="token_dense1")
    m.add_node(Activation('relu'),
                   name='token_relu1', input='token_dropout2')

    # add lemma nodes:
    m.add_node(Dense(3*dense_dims, nb_lemmas),
                   name='lemma_dense',
                   inputs=['left_relu1', 'token_dropout2', 'right_relu1'],
                   merge_mode='concat')
    m.add_node(Activation('softmax'),
                   name='lemma_softmax', input='lemma_dense')
    m.add_output(name='lemma_output', input='lemma_softmax')

    # add postag nodes:
    m.add_node(Dense(3*dense_dims, nb_postags),
                   name='pos_dense',
                   inputs=['left_relu1', 'token_dropout2', 'right_relu1'],
                   merge_mode='concat')
    m.add_node(Activation('softmax'),
                   name='pos_softmax', input='pos_dense')
    m.add_output(name='pos_output', input='pos_softmax')
    

    m.compile(optimizer='adagrad',
              loss={'lemma_output':'categorical_crossentropy',
                    'pos_output':'categorical_crossentropy'})

    return m


def build_lemmatizer_sequential(nb_filters,
                    std_token_len,
                    filter_length,
                    char_vector_dict,
                    nb_lemmas):
    
    token_model = Sequential()
    token_model.add(Convolution1D(input_dim=len(char_vector_dict),
                            nb_filter=nb_filters,
                            filter_length=filter_length,
                            activation="relu",
                            border_mode="valid",
                            subsample_length=1,
                            ))
    token_model.add(MaxPooling1D(pool_length=2))
    token_model.add(Flatten())
    token_model.add(Dropout(0.5))
    output_size =  nb_filters * (((std_token_len-filter_length)/1)+1)/2
    token_model.add(Dense(output_size, 500))
    token_model.add(Dropout(0.5))
    token_model.add(Activation('relu'))
    

    left_model = Sequential()
    left_model.add(Convolution1D(input_dim=len(char_vector_dict),
                            nb_filter=nb_filters,
                            filter_length=filter_length,
                            activation="relu",
                            border_mode="valid",
                            subsample_length=1,
                            ))
    left_model.add(MaxPooling1D(pool_length=2))
    left_model.add(LSTM(nb_filters/2, 500))
    left_model.add(Dropout(0.5))
    left_model.add(Activation('relu'))

    right_model = Sequential()
    right_model.add(Convolution1D(input_dim=len(char_vector_dict),
                            nb_filter=nb_filters,
                            filter_length=filter_length,
                            activation="relu",
                            border_mode="valid",
                            subsample_length=1,
                            ))
    right_model.add(MaxPooling1D(pool_length=2))
    right_model.add(LSTM(nb_filters/2, 500))
    right_model.add(Dropout(0.5))
    right_model.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([left_model, token_model, right_model], mode='concat'))
    model.add(Dense(1500, nb_lemmas))
    model.add(Activation('softmax'))

    #adagrad = Adagrad()
    model.compile(loss='categorical_crossentropy', optimizer="adagrad")
    return model

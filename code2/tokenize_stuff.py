#!usr/bin/env python

import numpy as np

from keras.optimizers import SGD, RMSprop, Adagrad
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D

from utils import get_char_vector_dict, vectorize_charseq

def vectorize(tokens, nb_left_tokens, left_char_len,
               nb_right_tokens, right_char_len, char_vector_dict=None):
    # unconcatenate the tokens
    unconcat_tokens, concat_y = [], []
    for token in tokens:
        # binarization of concatenation: whether (1) or not (0)
        # token should be appended to previous token:
        if "~" in token:
            comps = token.lower().split("~")
            # first comp shoudl'nt be appended:
            unconcat_tokens.append(comps[0])
            concat_y.append(0)
            # subsequent comps have to be appended:
            for comp in comps[1:]:
                unconcat_tokens.append(comp)
                concat_y.append(1)
        else:
            unconcat_tokens.append(token)
            concat_y.append(0)
    assert len(concat_y) == len(unconcat_tokens)
    
    if not char_vector_dict:
        char_vector_dict = get_char_vector_dict(unconcat_tokens)

    # create context vectors on either side of concatenation operation:
    left_X, right_X = [], []
    for token_idx, token in enumerate(unconcat_tokens):
        # vectorize left context:
        left_str = " ".join([unconcat_tokens[token_idx-(t+1)] for t in range(nb_left_tokens)
                             if token_idx-(t+1) >= 0])
        left_X.append(vectorize_charseq(left_str,
                            char_vector_dict, left_char_len))
        # vectorize right context:
        right_str = " ".join([unconcat_tokens[token_idx+t] for t in range(nb_right_tokens)
                             if token_idx+t < len(unconcat_tokens)])
        right_X.append(vectorize_charseq(right_str,
                            char_vector_dict, right_char_len))
    
    assert len(left_X) == len(right_X) == len(concat_y)
    left_X = np.asarray(left_X, dtype="int8")
    right_X = np.asarray(right_X, dtype="int8")
    return left_X, right_X, np.asarray(concat_y, dtype="int8"), char_vector_dict

def build_tokenizer(nb_filters = 1000,
                    filter_length = 3,
                    batch_size = 50,
                    char_vector_dict = {}):

    left_model = Sequential()
    left_model.add(Convolution1D(input_dim=len(char_vector_dict),
                            nb_filter=nb_filters,
                            filter_length=filter_length,
                            activation="relu",
                            border_mode="valid",
                            subsample_length=1,
                            ))
    left_model.add(MaxPooling1D(pool_length=2))
    left_model.add(LSTM(nb_filters/2, 250))
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
    right_model.add(LSTM(nb_filters/2, 250))
    right_model.add(Dropout(0.5))
    right_model.add(Activation('relu'))
    
    model = Sequential()
    model.add(Merge([left_model, right_model], mode='concat'))
    model.add(Dense(500, 1))
    model.add(Activation('sigmoid'))

    rms = RMSprop(clipnorm=0.3)
    model.compile(loss='binary_crossentropy', optimizer=rms, class_mode="binary")
    return model








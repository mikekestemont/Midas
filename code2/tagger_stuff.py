import numpy as np

from utils import get_char_vector_dict, encode_labels, vectorize_charseq

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.optimizers import Adagrad
from keras.models import Sequential
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

"""
def main():
    # define hyperparams:
    NB_INSTANCES = 50000
    STD_TOKEN_LEN = 12

    NB_LEFT_TOKENS = 2
    LEFT_CHAR_LEN = 21

    NB_RIGHT_TOKENS = 1
    RIGHT_CHAR_LEN = 10

    NB_FILTERS = 1000
    FILTER_LENGTH = 3

    BATCH_SIZE = 50

    tokens, postags, lemmas = load_data(file_path="../data/uniform/annotated/relig/train/relig_train.3col",
                                        nb_instances=NB_INSTANCES)


    char_vector_dict = get_char_vector_dict(tokens)
    
    
    lemmas = [lemma for lemma in lemmas if lemma not in ("$", "@")]
    lemmas_encoder, lemmas_y = encode_labels(lemmas)

    postags = [pos for pos in postags if pos not in ("$", "@")]
    pos_encoder, pos_y = encode_labels(postags)
    
    
    left_X, tokens_X, right_X = vectorize_tokens(tokens, char_vector_dict, STD_TOKEN_LEN,
                                        NB_LEFT_TOKENS, LEFT_CHAR_LEN,
                                        NB_RIGHT_TOKENS, RIGHT_CHAR_LEN,
                                        )
    print(tokens_X.shape)
    print(lemmas_y.shape)

    token_model = Sequential()
    token_model.add(Convolution1D(input_dim=len(char_vector_dict),
                            nb_filter=NB_FILTERS,
                            filter_length=FILTER_LENGTH,
                            activation="relu",
                            border_mode="valid",
                            subsample_length=1,
                            ))
    token_model.add(MaxPooling1D(pool_length=2))
    token_model.add(Flatten())
    output_size = NB_FILTERS * (((STD_TOKEN_LEN-FILTER_LENGTH)/1)+1)/2
    token_model.add(Dense(output_size, 250))
    token_model.add(Dropout(0.5))
    token_model.add(Activation('relu'))

    left_model = Sequential()
    left_model.add(Convolution1D(input_dim=len(char_vector_dict),
                            nb_filter=NB_FILTERS,
                            filter_length=FILTER_LENGTH,
                            activation="relu",
                            border_mode="valid",
                            subsample_length=1,
                            ))
    left_model.add(MaxPooling1D(pool_length=2))
    left_model.add(LSTM(NB_FILTERS/2, 250))
    left_model.add(Dropout(0.5))
    left_model.add(Activation('relu'))

    right_model = Sequential()
    right_model.add(Convolution1D(input_dim=len(char_vector_dict),
                            nb_filter=NB_FILTERS,
                            filter_length=FILTER_LENGTH,
                            activation="relu",
                            border_mode="valid",
                            subsample_length=1,
                            ))
    right_model.add(MaxPooling1D(pool_length=2))
    right_model.add(LSTM(NB_FILTERS/2, 250))
    right_model.add(Dropout(0.5))
    right_model.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([left_model, token_model, right_model], mode='concat'))
    model.add(Dense(750, len(lemmas_encoder.classes_)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    model.fit([left_X, tokens_X, right_X], lemmas_y, validation_split=0,
               batch_size=BATCH_SIZE, nb_epoch=50,
               show_accuracy=True, verbose=1)

if __name__ == "__main__":
    main()
"""










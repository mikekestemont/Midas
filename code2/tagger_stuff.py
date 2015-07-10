import numpy as np

from sklearn.preprocessing import LabelEncoder

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Convolution1D, MaxPooling1D

def vectorize_tokens(tokens, char_vector_dict, std_token_len,
                     nb_left_tokens, left_char_len,
                     nb_right_tokens, right_char_len,
                     ):
    left_X, tokens_X, right_X = [], [], []
    for token_idx, token in enumerate(tokens):
        # ignore boundari markers:
        if token in ("@", "$"):
            continue
        # vectorize target token:
        tokens_X.append(vectorize_charseq(token,
                            char_vector_dict, std_token_len))
        # vectorize left context:
        left_str = " ".join([tokens[token_idx-(t+1)] for t in range(nb_left_tokens)
                             if token_idx-(t+1) >= 0])
        left_X.append(vectorize_charseq(left_str,
                            char_vector_dict, left_char_len))
        # vectorize right context:
        right_str = " ".join([tokens[token_idx+t+1] for t in range(nb_right_tokens)
                             if token_idx+t+1 < len(tokens)])
        right_X.append(vectorize_charseq(right_str,
                            char_vector_dict, right_char_len))
    assert len(tokens_X) == len(left_X)
    assert len(tokens_X) == len(right_X)
    tokens_X = np.asarray(tokens_X, dtype="int8")
    left_X = np.asarray(left_X, dtype="int8")
    right_X = np.asarray(right_X, dtype="int8")
    return left_X, tokens_X, right_X

def encode_labels(labels):
    label_encoder = LabelEncoder()
    labels_y = label_encoder.fit_transform(labels)
    labels_y = np_utils.to_categorical(labels_y, len(label_encoder.classes_))
    return label_encoder, labels_y


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










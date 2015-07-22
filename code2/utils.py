import numpy as np

from keras.utils import np_utils

def get_char_vector_dict(tokens):
    char_vocab = tuple({ch for tok in tokens for ch in tok+" "})
    char_vector_dict = {}
    filler = np.zeros(len(char_vocab), dtype="int8")
    for idx, char in enumerate(char_vocab):
        ph = filler.copy()
        ph[idx] = 1
        char_vector_dict[char] = ph
    return char_vector_dict

def vectorize_charseq(seq, char_vector_dict, std_seq_len):
    seq_X = []
    filler = np.zeros(len(char_vector_dict), dtype="int8")

    # cut, if needed:
    seq = seq[:std_seq_len]

    for char in seq:
        try:
            seq_X.append(char_vector_dict[char])
        except KeyError:
            seq_X.append(filler)

    # pad, if needed:
    while len(seq_X) < std_seq_len:
        seq_X.append(filler)

    return np.vstack(seq_X)

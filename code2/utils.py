import numpy as np

def get_char_vector_dict(tokens):
    char_vocab = tuple({ch for tok in tokens for ch in tok+" "})
    char_vector_dict = {}
    for idx, char in enumerate(char_vocab):
        filler = np.zeros(len(char_vocab), dtype="int8")
        filler[idx] = 1
        char_vector_dict[char] = filler
    return char_vector_dict

def vectorize_charseq(seq, char_vector_dict, std_seq_len):
    seq_X = []
    filler = np.zeros(len(char_vector_dict), dtype="int8")
    # cut, if needed:
    seq = seq[:std_seq_len]
    for idx, char in enumerate(seq):
        try:
            seq_X.append(char_vector_dict[char])
        except KeyError:
            seq_X.append(filler)
    # pad, if needed:
    while len(seq_X) < std_seq_len:
        seq_X.append(filler)
    return np.asarray(seq_X)
#!usr/bin/env python

import os
import glob

def load_annotated_file(filepath, nb_instances=5000):
    """
    Parses an annotated data file
    Eats:
        - the filepath
        - nb_instances, max nb of instances to load (for dev purposes)
    Spits: all the annotated instances as three lists:
           tokens, postags, lemmas
           Notes:
           - The content of the files under dir will be concatenated
           - The tokens will contain the concatenated tokens (with "~")
    """
    tokens, postags, lemmas = [], [], []
    with open(filepath, 'r') as data_file:
        for line in data_file:
            line = line.strip()
            if not line:
                tok, pos, lem = "$", "$", "$" # mark beginning of utterances
            elif line.startswith("@"):
                tok, pos, lem = "@", "@", "@" # mark beginning of documents
            else:
                tok, pos, lem = line.strip().split("\t")
            tokens.append(tok)
            postags.append(pos)
            lemmas.append(lem)
            if len(tokens) >= nb_instances:
                return tokens, postags, lemmas
    return tokens, postags, lemmas


def load_annotated_data_dir(data_dir, nb_instances=10000):
    """
    Parses the list of annotated data files in a directory
    Eats:
        - data_dir, the path to a directory
        - nb_instances, max nb of instances to load from a file (for dev purposes)
    Spits: all the annotated instances as three tuples:
           tokens, postags, lemmas
           Notes:
           - The content of the files under dir will be concatenated
           - The tokens will contain the concatenated tokens (with "~")
    """
    tokens, postags, lemmas = [], [], []
    if os.path.isdir(data_dir):
        for filepath in glob.glob(data_dir+"/*"):
            ts, ps, ls = load_annotated_file(filepath, nb_instances)
            tokens.extend(ts)
            postags.extend(ps)
            lemmas.extend(ls)
    return tuple(tokens), tuple(postags), tuple(lemmas)

#print(load_annotated_data("../data/uniform/annotated/cg-lit/train"))





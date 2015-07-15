"""
    Usage:
       >>> python midas.py config.txt
"""

from __future__ import print_function

import sys
import os
import shutil
import cPickle as pickle

import cmd_line
import datasets

import tokenize_stuff
import tagger_stuff
import utils

from keras.utils import np_utils

BATCH_SIZE = 50

def main():
    print("::: midas started :::")

    param_dict = dict()
    config_path = os.path.abspath(sys.argv[1])
    print("> using config file: "+str(config_path))

    if config_path:
        param_dict = cmd_line.parse(config_path)
    else:
        raise ValueError("No config file specified.")
    #print(param_dict)

    model_name = param_dict["model_name"]

    if param_dict["mode"] == "train":

        print("> start training")

        # make sure that we have a fresh model folder to work in
        if not os.path.isdir("../models"):
            os.mkdir("../models")
        if os.path.isdir("../models/"+model_name):
            shutil.rmtree("../models/"+model_name)
        os.mkdir("../models/"+model_name)

        train_tokens, train_postags, train_lemmas = \
            datasets.load_annotated_data_dir(data_dir = os.path.abspath(param_dict["train_dir"]),
                                         nb_instances = 150000)

        if param_dict["tokenize"]:
            left_X, right_X, concat_y, char_vector_dict = \
                tokenize_stuff.vectorize(tokens = train_tokens,
                                         nb_left_tokens = param_dict["tok_nb_left_tokens"],
                                         left_char_len = param_dict["tok_left_char_len"],
                                         nb_right_tokens = param_dict["tok_nb_right_tokens"],
                                         right_char_len = param_dict["tok_right_char_len"])
            
            tokenizer = tokenize_stuff.build_tokenizer(nb_filters = 2500,
                                        filter_length = 3,
                                        char_vector_dict = char_vector_dict)

            tokenizer.fit([left_X, right_X], concat_y, validation_split = 0.20, show_accuracy=True,
                            batch_size = BATCH_SIZE, nb_epoch = param_dict["tok_nb_epochs"], class_weight={0:1, 1:100})

            # save relevant objects:
            tokenizer.save_weights("../models/"+model_name+"/tokenizer.model_weights",
                                    overwrite=True)
            pickle.dump(char_vector_dict,
                         open("../models/"+model_name+"/char_vector_dict.p", "wb" ))

        if param_dict["postag"]:
            pass

        if param_dict["lemmatize"]:

            train_lemmas = [lem for lem in train_lemmas if lem not in ("@", "$")]
            train_postags = [pos for pos in train_postags if pos not in ("@", "$")]

            lemma_encoder, train_lemmas_y = utils.encode_labels(train_lemmas)
            pos_encoder, train_pos_y = utils.encode_labels(train_postags)

            print(len(train_lemmas_y))
            print(len(train_pos_y))

            lemma_labels_y = np_utils.to_categorical(train_lemmas_y, len(lemma_encoder.classes_))
            pos_labels_y = np_utils.to_categorical(train_pos_y, len(pos_encoder.classes_))

            left_X, tokens_X, right_X, char_vector_dict = tagger_stuff.vectorize(tokens = train_tokens,
                                        std_token_len = param_dict["lemma_std_len_token"],
                                        nb_left_tokens = param_dict["lemma_nb_left_tokens"],
                                        left_char_len = param_dict["lemma_left_char_len"],
                                        nb_right_tokens = param_dict["lemma_nb_right_tokens"],
                                        right_char_len = param_dict["lemma_right_char_len"],
                                        )
            print(tokens_X.shape)

            lemmatizer = tagger_stuff.build_lemmatizer(nb_filters = 1000,
                                        filter_length = 3,
                                        std_token_len = param_dict["lemma_std_len_token"],
                                        char_vector_dict = char_vector_dict,
                                        nb_lemmas = len(lemma_encoder.classes_),
                                        nb_postags = len(pos_encoder.classes_),
                                        dense_dims = 250,
                                        )
            for e in range(param_dict["lemma_nb_epochs"]):
                print("-> epoch ", e+1, "...")
                lemmatizer.fit({'left_input': left_X,
                                          'token_input': tokens_X,
                                          'right_input': right_X,
                                          'lemma_output': lemma_labels_y,
                                          'pos_output': pos_labels_y
                                         },
                                nb_epoch = 1,
                                batch_size = BATCH_SIZE)
                predictions = lemmatizer.predict({'left_input': left_X,
                                          'token_input': tokens_X,
                                          'right_input': right_X,
                                         },
                                batch_size = BATCH_SIZE)
                pos_predictions = np_utils.categorical_probas_to_classes(predictions['pos_output'])
                pos_accuracy = np_utils.accuracy(pos_predictions, train_pos_y)
                print("\t - postags acc:\t{:.2%}".format(pos_accuracy))
                lemma_predictions = np_utils.categorical_probas_to_classes(predictions['lemma_output'])
                lemma_accuracy = np_utils.accuracy(lemma_predictions, train_lemmas_y)
                print("\t - lemmas acc:\t{:.2%}".format(lemma_accuracy))
            #lemmatizer.fit([left_X, tokens_X, right_X], labels_y, show_accuracy=True,
            #                batch_size = BATCH_SIZE, nb_epoch = param_dict["lemma_nb_epochs"])
            #lemmatizer.fit(tokens_X, labels_y, show_accuracy=True,
            #                batch_size = 50, nb_epoch = param_dict["lemma_nb_epochs"])

    elif param_dict["mode"] == "test":
        
        print("> start testing")

        test_tokens, test_postags, test_lemmas = \
            datasets.load_annotated_data_dir(data_dir = os.path.abspath(param_dict["input_dir"]),
                                         nb_instances = 5000)
        char_vector_dict = pickle.load(open("../models/"+model_name+"/char_vector_dict.p", "rb"))
        left_X, right_X, concat_y, _ = \
                tokenize_stuff.vectorize(tokens = test_tokens,
                                         nb_left_tokens = param_dict["tok_nb_left_tokens"],
                                         left_char_len = param_dict["tok_left_char_len"],
                                         nb_right_tokens = param_dict["tok_nb_right_tokens"],
                                         right_char_len = param_dict["tok_right_char_len"],
                                         char_vector_dict = char_vector_dict)
        tokenizer = tokenize_stuff.build_tokenizer(nb_filters = 250,
                                        filter_length = 3,
                                        char_vector_dict = char_vector_dict)
        tokenizer.load_weights("../models/"+model_name+"/tokenizer.model_weights")
        preds = tokenizer.predict_classes([left_X, right_X], batch_size = 1000)
        for item in zip(tokenize_stuff.unconcatenate_tokens(test_tokens)[0], preds):
            print(item)
        
        



    print("::: midas ended :::")


if __name__ == "__main__":
    main()

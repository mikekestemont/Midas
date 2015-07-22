"""
    Usage:
       >>> python midas.py config.txt
"""

from __future__ import print_function

import sys
import os
import shutil
import cPickle as pickle

from collections import Counter

import cmd_line
import datasets

import tokenize_stuff
import tagger_stuff
import utils

from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder

BATCH_SIZE = 100

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
                                         nb_instances = 500000000000)
        dev_tokens, dev_postags, dev_lemmas = \
            datasets.load_annotated_data_dir(data_dir = os.path.abspath(param_dict["dev_dir"]),
                                         nb_instances = 500000000000)

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
            train_labels = [lem+"_"+pos for lem, pos in zip(train_lemmas, train_postags)]

            dev_lemmas = [lem for lem in dev_lemmas if lem not in ("@", "$")]
            dev_postags = [pos for pos in dev_postags if pos not in ("@", "$")]
            dev_labels = [lem+"_"+pos for lem, pos in zip(dev_lemmas, dev_postags)]

            label_encoder = LabelEncoder()
            label_encoder.fit(train_labels+dev_labels)
            train_ints = label_encoder.transform(train_labels)
            dev_ints = label_encoder.transform(dev_labels)

            train_y = np_utils.to_categorical(train_ints, len(label_encoder.classes_))
            dev_y = np_utils.to_categorical(dev_ints, len(label_encoder.classes_))

            train_left_X, train_tokens_X, train_right_X, train_char_vector_dict = tagger_stuff.vectorize(tokens = train_tokens,
                                        std_token_len = param_dict["lemma_std_len_token"],
                                        nb_left_tokens = param_dict["lemma_nb_left_tokens"],
                                        left_char_len = param_dict["lemma_left_char_len"],
                                        nb_right_tokens = param_dict["lemma_nb_right_tokens"],
                                        right_char_len = param_dict["lemma_right_char_len"],
                                        )
            print(train_tokens_X.shape)
            
            dev_left_X, dev_tokens_X, dev_right_X, _ = tagger_stuff.vectorize(tokens = dev_tokens,
                                        std_token_len = param_dict["lemma_std_len_token"],
                                        nb_left_tokens = param_dict["lemma_nb_left_tokens"],
                                        left_char_len = param_dict["lemma_left_char_len"],
                                        nb_right_tokens = param_dict["lemma_nb_right_tokens"],
                                        right_char_len = param_dict["lemma_right_char_len"],
                                        char_vector_dict = train_char_vector_dict
                                        )
            print(dev_tokens_X.shape)

            lemmatizer = tagger_stuff.build_lemmatizer_new(nb_filters = 1024,
                                        filter_length = 3,
                                        std_token_len = param_dict["lemma_std_len_token"],
                                        left_char_len = param_dict["lemma_left_char_len"],
                                        right_char_len = param_dict["lemma_right_char_len"],
                                        char_vector_dict = train_char_vector_dict,
                                        nb_labels = len(label_encoder.classes_),
                                        dense_dims = 1024,
                                        )

            for e in range(param_dict["lemma_nb_epochs"]):
                print("-> epoch ", e+1, "...")
                lemmatizer.fit({#'left_input': train_left_X,
                                          'token_input': train_tokens_X,
                                          #'right_input': train_right_X,
                                          'label_output': train_y
                                         },
                                nb_epoch = 1,
                                batch_size = BATCH_SIZE)

                print("+++ TRAIN SCORE")
                predictions = lemmatizer.predict({#'left_input': train_left_X,
                                          'token_input': train_tokens_X,
                                          #'right_input': train_right_X,
                                         },
                                batch_size = BATCH_SIZE)
                predictions = np_utils.categorical_probas_to_classes(predictions['label_output'])
                accuracy = np_utils.accuracy(predictions, train_ints)
                print("\t - acc:\t{:.2%}".format(accuracy))

                print("+++ DEV SCORE")
                predictions = lemmatizer.predict({#'left_input': dev_left_X,
                                          'token_input': dev_tokens_X,
                                          #'right_input': dev_right_X,
                                         },
                                batch_size = BATCH_SIZE)
                predictions = np_utils.categorical_probas_to_classes(predictions['label_output'])
                accuracy = np_utils.accuracy(predictions, dev_ints)
                print("\t - acc:\t{:.2%}".format(accuracy))
            """
            ########################################################################################################
            # train data:
            train_lemmas = [lem for lem in train_lemmas if lem not in ("@", "$")]
            train_postags = [pos for pos in train_postags if pos not in ("@", "$")]
            print("orig nb lemmas:", len(set(train_lemmas)))

            train_lemma_counter = Counter(train_lemmas)
            train_lemma_vocab = [k for k, v in train_lemma_counter.items() if v > 1]
            print("reduced nb lemmas:", len(train_lemma_vocab))

            train_lemmas = [lem if lem in train_lemma_vocab else '<unk>' for lem in train_lemmas]

            dev_lemmas = [lem for lem in dev_lemmas if lem not in ("@", "$")]
            dev_postags = [pos for pos in dev_postags if pos not in ("@", "$")]

            dev_lemmas = [lem if lem in train_lemma_vocab else '<unk>' for lem in dev_lemmas]

            lemma_encoder = LabelEncoder()
            lemma_encoder.fit(train_lemmas+dev_lemmas+['<unk>'])
            train_lemmas_y = lemma_encoder.transform(train_lemmas)
            dev_lemmas_y = lemma_encoder.transform(dev_lemmas)

            pos_encoder = LabelEncoder()
            pos_encoder.fit(train_postags+dev_postags)
            train_pos_y = pos_encoder.transform(train_postags)
            dev_pos_y = pos_encoder.transform(dev_postags)

            train_lemma_labels_y = np_utils.to_categorical(train_lemmas_y, len(lemma_encoder.classes_))
            train_pos_labels_y = np_utils.to_categorical(train_pos_y, len(pos_encoder.classes_))

            dev_lemma_labels_y = np_utils.to_categorical(dev_lemmas_y, len(lemma_encoder.classes_))
            dev_pos_labels_y = np_utils.to_categorical(dev_pos_y, len(pos_encoder.classes_))

            train_left_X, train_tokens_X, train_right_X, train_char_vector_dict = tagger_stuff.vectorize(tokens = train_tokens,
                                        std_token_len = param_dict["lemma_std_len_token"],
                                        nb_left_tokens = param_dict["lemma_nb_left_tokens"],
                                        left_char_len = param_dict["lemma_left_char_len"],
                                        nb_right_tokens = param_dict["lemma_nb_right_tokens"],
                                        right_char_len = param_dict["lemma_right_char_len"],
                                        )
            print(train_tokens_X.shape)
            
            dev_left_X, dev_tokens_X, dev_right_X, _ = tagger_stuff.vectorize(tokens = dev_tokens,
                                        std_token_len = param_dict["lemma_std_len_token"],
                                        nb_left_tokens = param_dict["lemma_nb_left_tokens"],
                                        left_char_len = param_dict["lemma_left_char_len"],
                                        nb_right_tokens = param_dict["lemma_nb_right_tokens"],
                                        right_char_len = param_dict["lemma_right_char_len"],
                                        char_vector_dict = train_char_vector_dict
                                        )
            print(dev_tokens_X.shape)

            lemmatizer = tagger_stuff.build_lemmatizer(nb_filters = 1024,
                                        filter_length = 3,
                                        std_token_len = param_dict["lemma_std_len_token"],
                                        left_char_len = param_dict["lemma_left_char_len"],
                                        right_char_len = param_dict["lemma_right_char_len"],
                                        char_vector_dict = train_char_vector_dict,
                                        nb_lemmas = len(lemma_encoder.classes_),
                                        nb_postags = len(pos_encoder.classes_),
                                        dense_dims = 500
                                        )

            for e in range(param_dict["lemma_nb_epochs"]):
                print("-> epoch ", e+1, "...")
                lemmatizer.fit({'left_input': train_left_X,
                                          'token_input': train_tokens_X,
                                          'right_input': train_right_X,
                                          'lemma_output': train_lemma_labels_y,
                                          'pos_output': train_pos_labels_y
                                         },
                                nb_epoch = 1,
                                batch_size = BATCH_SIZE)

                print("+++ TRAIN SCORE")
                predictions = lemmatizer.predict({'left_input': train_left_X,
                                          'token_input': train_tokens_X,
                                          'right_input': train_right_X,
                                         },
                                batch_size = BATCH_SIZE)
                pos_predictions = np_utils.categorical_probas_to_classes(predictions['pos_output'])
                pos_accuracy = np_utils.accuracy(pos_predictions, train_pos_y)
                print("\t - postags acc:\t{:.2%}".format(pos_accuracy))
                lemma_predictions = np_utils.categorical_probas_to_classes(predictions['lemma_output'])
                lemma_accuracy = np_utils.accuracy(lemma_predictions, train_lemmas_y)
                print("\t - lemmas acc:\t{:.2%}".format(lemma_accuracy))

                print("+++ DEV SCORE")
                dev_predictions = lemmatizer.predict({'left_input': dev_left_X,
                                          'token_input': dev_tokens_X,
                                          'right_input': dev_right_X,
                                         },
                                batch_size = BATCH_SIZE)
                dev_pos_predictions = np_utils.categorical_probas_to_classes(dev_predictions['pos_output'])
                dev_pos_accuracy = np_utils.accuracy(dev_pos_predictions, dev_pos_y)
                print("\t - postags acc:\t{:.2%}".format(dev_pos_accuracy))
                dev_lemma_predictions = np_utils.categorical_probas_to_classes(dev_predictions['lemma_output'])
                dev_lemma_accuracy = np_utils.accuracy(dev_lemma_predictions, dev_lemmas_y)
                print("\t - lemmas acc:\t{:.2%}".format(dev_lemma_accuracy))
        """

    elif param_dict["mode"] == "test":
        
        print("> start testing")

        test_tokens, test_postags, test_lemmas = \
            datasets.load_annotated_data_dir(data_dir = os.path.abspath(param_dict["test_dir"]),
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

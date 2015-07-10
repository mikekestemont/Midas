"""
    Usage:
       >>> python midas.py config.txt
"""

import sys
import os
import shutil
import cPickle as pickle

import cmd_line
import datasets

import tokenize_stuff

BATCH_SIZE = 30

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
                                         nb_instances = 2000)

        if param_dict["tokenize"]:
            left_X, right_X, concat_y, char_vector_dict = \
                tokenize_stuff.vectorize(tokens = train_tokens,
                                         nb_left_tokens = param_dict["tok_nb_left_tokens"],
                                         left_char_len = param_dict["tok_left_char_len"],
                                         nb_right_tokens = param_dict["tok_nb_right_tokens"],
                                         right_char_len = param_dict["tok_right_char_len"])
            
            tokenizer = tokenize_stuff.build_tokenizer(nb_filters = 1000,
                                        filter_length = 3,
                                        batch_size = 50,
                                        char_vector_dict = char_vector_dict)

            tokenizer.fit([left_X, right_X], concat_y, validation_split = 0.20,
                            batch_size = BATCH_SIZE, nb_epoch = param_dict["tok_nb_epochs"])

            # save relevant objects:
            tokenizer.save_weights("../models/"+model_name+"/tokenizer.model_weights",
                                    overwrite=True)
            pickle.dump(char_vector_dict,
                         open("../models/"+model_name+"/char_vector_dict.p", "wb" ))

        if param_dict["postag"]:
            pass

        if param_dict["lemmatize"]:
            pass

    elif param_dict["mode"] == "test":
        
        print("> start testing")

        test_tokens, test_postags, test_lemmas = \
            datasets.load_annotated_data_dir(data_dir = os.path.abspath(param_dict["input_dir"]),
                                         nb_instances = 2000)
        char_vector_dict = pickle.load(open("../models/"+model_name+"/char_vector_dict.p", "rb"))
        left_X, right_X, concat_y, _ = \
                tokenize_stuff.vectorize(tokens = test_tokens,
                                         nb_left_tokens = param_dict["tok_nb_left_tokens"],
                                         left_char_len = param_dict["tok_left_char_len"],
                                         nb_right_tokens = param_dict["tok_nb_right_tokens"],
                                         right_char_len = param_dict["tok_right_char_len"],
                                         char_vector_dict = char_vector_dict)
        tokenizer = tokenize_stuff.build_tokenizer(nb_filters = 1000,
                                        filter_length = 3,
                                        batch_size = 50,
                                        char_vector_dict = char_vector_dict)
        tokenizer.load_weights("../models/"+model_name+"/tokenizer.model_weights")
        tokenizer.predict([left_X, right_X], batch_size = BATCH_SIZE)
        
        



    print("::: midas ended :::")


if __name__ == "__main__":
    main()

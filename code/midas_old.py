#!usr/bin/env python

# dependencies: sklearn, maxent, py-editdist (better to download this from google code than github; didn't compile on the cals)

import sys
import ConfigParser
import os
import re
import codecs

from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

from Gazetteer import *
from Tokenizer import *
from MaxentTagger import *

TAB = re.compile(r"\s*\t\s*")

def load_training_data(trainingD_path=""):
    print("Loading training data from directory: %s") % trainingD_path
    training_items = []
    for filename in os.listdir(trainingD_path):
        if filename.endswith((".txt", "3col")):
            with open(os.sep.join([trainingD_path, filename]), 'r') as infile:
                for line in infile.readlines():#[:30000]:
                    line = line.strip()
                    if not line:
                        training_items.append(("<utt>", "<utt>", "<utt>"))
                    elif not line.startswith("@"): # ignore metadata
                        items = TAB.split(line)
                        try:
                            token, pos, lemma = items
                            training_items.append((token, pos, lemma))
                        except:
                            pass
            print("\t* Loaded %s") % (filename)
    print("Loaded %s training items") % (len(training_items))
    return training_items

def train(training_items=[],
          WS="",
          gazetteer=None,
          tokenize=False,
          tokenizer_context=1,
          left_context=1,
          right_context=1,
          alignment_context=1,
          max_align_dist=1,
          nearest_lev_hyperballs=1,
          max_lev_dist=1,
          min_tok_freq=25,
          min_lem_freq=25):
    """Train and save a new model"""
    ####### train tokenizer ####################################################################################
    if tokenize:
        tokenizer = Tokenizer(context=tokenizer_context,
                              WS=WS)
        tokenizer.train(training_items=training_items,
                        gazetteer=gazetteer)
    ####### train POS-tagger and lemmatizer ################################################################################################
    sequential_tagger = MaxentTagger(WS=WS,
                                     left_context=left_context,
                                     right_context=right_context,
                                     min_tok_freq=min_tok_freq,
                                     min_lem_freq=min_lem_freq)
    sequential_tagger.train(training_items=training_items,
                            gazetteer=gazetteer)
    sequential_tagger.train_lemmatizer(training_items=training_items,
                                       alignment_context=alignment_context,
                                       max_align_dist=max_align_dist,
                                       nearest_lev_hyperballs=nearest_lev_hyperballs,
                                       max_lev_dist=max_lev_dist)
    return

def load_test_data(filepath="", mode=""):
    items = []
    if mode == "tag":
        text = codecs.open(filepath, 'r', encoding="utf-8").read().lower()
        text = "".join([char for char in text if (char.isalpha() or char.isspace())])
        items = str(text).split()
    elif mode == "test":
        with open(filepath, 'r') as infile:
            for line in infile.readlines():
                line = line.strip()
                if not line:
                    items.append("<utt>")
                elif not line.startswith("@"): # ignore metadata
                    try:
                        token, pos, lemma = TAB.split(line)
                        items.append([token, pos, lemma])
                    except:
                        pass
            print("\t* Loaded %s") % (filepath)
    else:
        raise ValueError("Something wrong with mode parameter...")
    return items

def tag(test_items=[],
        WS="",
        tokenize=False,
        tokenizer_context=0,
        left_context=1,
        right_context=1,
        gazetteer=None,
        min_tok_freq=25,
        min_lem_freq=25,
        mode=""):
    """
    Tags a list of (potentially annotated) test tokens.
    """
    tokenized_tokens = []
    token_acc, token_f1 = None, None
    if tokenize:
        # load and apply a tokenizer:
        tokenizer = Tokenizer(context=tokenizer_context, WS=WS)
        if mode == "tag":
            tokenized_tokens = tokenizer.tokenize(test_items=test_items, gazetteer=gazetteer)    
        elif mode in ("test", "crossval"):
            items = []
            for item in test_items:
                if item == "<utt>":
                    items.append(item)
                else:
                    items.append(item[0].lower())
            token_acc, token_f1 = tokenizer.eval_tokenizer(test_items=items, gazetteer=gazetteer)
            # return the original tokens since we only tokenize for evaluation purposes:
            tokenized_tokens = items
    else:
        # assume the input has been properly tokenized already:
        if mode == "tag":
          tokenized_tokens = test_items
        elif mode == "test":
          tokenized_tokens = tuple(item[0].lower() for item in test_items)
    sequential_tagger = MaxentTagger(WS=WS, 
                                     left_context=left_context,
                                     right_context=right_context,
                                     min_tok_freq=min_tok_freq,
                                     min_lem_freq=min_lem_freq)
    sequential_tagger.load_models()
    tagged_items = sequential_tagger.tag(tokenized=tokenized_tokens, gazetteer=gazetteer)
    if mode in ("crossval", "test"):
        results = sequential_tagger.evaluate_tags_and_lemmas(gold_items=test_items,\
                                                             silver_items=tagged_items)
        if tokenize:
            results.extend((token_acc, token_f1))
        return results
    else:
        return tagged_items

def tag_dir(testD_path="",
            outputD_path="",
            WS="",
            tokenize=False,
            tokenizer_context=0,
            left_context=1,
            right_context=1,
            gazetteer=None,
            min_tok_freq=25,
            min_lem_freq=25,
            mode=""):
    print("Loading from "+testD_path+": ")
    for filename in os.listdir(testD_path):
        if filename.endswith((".txt", ".3col")):
            print("\t* "+filename)
            test_items = load_test_data(os.sep.join([testD_path, filename]), mode=mode)
            results = tag(test_items=test_items,
                          WS=WS,
                          tokenize=tokenize,
                          tokenizer_context=tokenizer_context,
                          left_context=left_context,
                          right_context=right_context,
                          gazetteer=gazetteer,
                          min_tok_freq=min_tok_freq,
                          min_lem_freq=min_lem_freq,
                          mode=mode)
            if mode == "tag":
                # simply write away the output:
                with open(os.sep.join([outputD_path, filename+".tag"]), "w+") as outputfile:
                    for token, pos, lemma in results:
                        if token != "<utt>":
                            outputfile.write("\t".join([token, pos, lemma])+"\n")
                        else:
                            outputfile.write("\n")
            elif mode == "test":
                # print the evaluation results
                if tokenize:
                    nr, nr_seen, nr_unseen,\
                    corr_pos_seen, corr_pos_unseen, corr_pos,\
                    corr_lem_seen, corr_lem_unseen, corr_lem,\
                    token_acc, token_f1 = results
                else:
                    nr, nr_seen, nr_unseen,\
                    corr_pos_seen, corr_pos_unseen, corr_pos,\
                    corr_lem_seen, corr_lem_unseen, corr_lem = results
                test_info = "=================== TEST RESULTS ===================\n"
                test_info += "Number of tokens: %s tokens\n" % (int(nr))
                test_info += "Proportion of seen tokens: %s%%\n" % (nr_seen*100)
                test_info += "Proportion of unseen tokens: %s%%\n" % (nr_unseen*100)
                test_info += "Correct pos: %s%%\n" % (corr_pos*100)
                test_info += "\t* Correct pos seen tokens: %s%%\n" % (corr_pos_seen*100)
                test_info += "\t* Correct pos unseen tokens: %s%%\n" % (corr_pos_unseen*100)
                test_info += "Correct lemma: %s%%\n" % (corr_lem*100)
                test_info += "\t* Correct lemma seen tokens: %s%%\n" % (corr_lem_seen*100)
                test_info += "\t* Correct lemma unseen tokens: %s%%\n" % (corr_lem_unseen*100)
                if tokenize:
                    test_info += "\t* Tokenizer accuracy: %s%%\n" % (token_acc*100)
                    test_info += "\t* Tokenizer f1-score: %s%%\n" % (token_f1*100)
                test_info += "========================================================"
                print(test_info)
    return
###################################################################################################
# execution starts here:
# parse command-line options: e.g. python Midas.py tag config.txt LITERARY
print("::: midas started :::")
mode, config_path, model_name = "", "", ""
testD_path, outputD_path = "", ""
try:
    if len(sys.argv[1:]) == 3:
        mode, config_path, model_name = sys.argv[1:]
        print("\t- mode: %s") % mode # "train", "tag" (raw input data), "crossval" (cross-validate on input data), "test" (test on annotated data)
        print("\t- config file: %s") % config_path # path to the config file
        print("\t- model name: %s") % model_name # the name model to load/train
    elif len(sys.argv[1:]) == 5:
        # for use in the online interface
        mode, config_path, model_name, testD_path, outputD_path = sys.argv[1:]
        print("\t- mode: %s") % mode # "train", "tag" (raw input data), "crossval" (cross-validate on input data), "test" (test on annotated data)
        print("\t- config file: %s") % config_path # path to the config file
        print("\t- model name: %s") % model_name # the name model to load/train
        print("\t- input dir: %s") % testD_path # the name model to load/train
        print("\t- output dir: %s") % outputD_path # the folder where the output has to be written
except ValueError:
    raise ValueError("Something wrong with the command-line parameters specified...")
#########################################################
# parse the configuration file:
config = ConfigParser.ConfigParser()
config.read(config_path)
# global options:
trainingD_path = config.get("global", "trainingD_path")
if not testD_path and not outputD_path: # only load these params from config.txt if they have not been specified on the command line
    testD_path = config.get("global", "testD_path")
    outputD_path = config.get("global", "outputD_path")
WS = config.get("global", "WS")
nr_folds = config.getint("global", "nr_folds")
# gazetteer options:
use_gazetteer = config.getboolean("gazetteer", "use_gazetteer")
path2gazetteers = config.get("gazetteer", "path2gazetteers")
# tokenizer options:
tokenize = config.getboolean("tokenizer", "tokenize")
tokenizer_context = config.getint("tokenizer", "tokenizer_context")
# tagger options:
left_context = config.getint("tagger", "left_context")
right_context = config.getint("tagger", "right_context")
min_lem_freq = config.getint("tagger", "min_lem_freq")
min_tok_freq = config.getint("tagger", "min_tok_freq")
# levenshtein options:
nearest_lev_hyperballs = config.getint("levenshtein", "nearest_lev_hyperballs")
max_lev_dist = config.getint("levenshtein", "max_lev_dist")
# alignator options:
alignment_context = config.getint("alignator", "alignment_context")
max_align_dist = config.getint("alignator", "max_align_dist")
#########################################################
# load the gazetteer if necessary:
gazetteer = None
if use_gazetteer:
    gazetteer = Gazetteer(path2gazetteers=path2gazetteers)
# determine path where to save the model dir:
if not os.path.isdir(WS):
  os.mkdir(WS)
if mode == "crossval":
  model_name = "crossval"
# expand the path using the model name
WS = os.sep.join([WS,model_name,""])
#########################################################
if mode == "train":
    training_items = load_training_data(trainingD_path)
    train(training_items=training_items,
          WS=WS,
          gazetteer=gazetteer,
          tokenize=tokenize,
          tokenizer_context=tokenizer_context,
          left_context=left_context,
          right_context=right_context,
          alignment_context=alignment_context,
          max_align_dist=max_align_dist,
          nearest_lev_hyperballs=nearest_lev_hyperballs,
          max_lev_dist=max_lev_dist,
          min_tok_freq = min_tok_freq,
          min_lem_freq=min_lem_freq)
##########################################################
elif mode == "tag" or mode == "test":
    tag_dir(testD_path=testD_path,
            outputD_path=outputD_path,
            WS=WS,
            tokenize=tokenize,
            tokenizer_context=tokenizer_context,
            left_context=left_context,
            right_context=right_context,
            gazetteer=gazetteer,
            min_tok_freq = min_tok_freq,
            min_lem_freq=min_lem_freq,
            mode=mode)
elif mode == "crossval":
    # set evaluation scores for folds:
    nr, nr_seen, nr_unseen = [], [], []
    corr_pos_seen, corr_pos_unseen, corr_pos = [], [], []
    corr_lem_seen, corr_lem_unseen, corr_lem = [], [], []
    if tokenize:
        token_acc, token_f1 = [], []
    # load all training items
    training_items = load_training_data(trainingD_path)
    # get cross-validation indices via sklearn:
    fold_indices = cross_validation.KFold(n=len(training_items),
                                          n_folds=nr_folds,\
                                          indices=True,\
                                          shuffle=False)
    current_fold_nr = 0
    cv_scores = []
    # start the cross-validation:
    for training_indices, test_indices in fold_indices:
        current_fold_nr+=1
        print("\t::::::::: Fold %d ::::::::::::::") % (current_fold_nr)
        # extract training items for current fold:
        fold_training_items = [training_items[index] for index in training_indices]
        print("\t\t- Started processing %d training items in fold %d...") % (len(fold_training_items), current_fold_nr)
        # extract test items for current fold:
        fold_test_items = [training_items[index] for index in test_indices]
        print("\t\t- Started processing %d test items in fold %d...") % (len(fold_test_items), current_fold_nr)
        train(training_items=fold_training_items,
              WS=WS,
              gazetteer=gazetteer,
              tokenize=tokenize,
              tokenizer_context=tokenizer_context,
              left_context=left_context,
              right_context=right_context,
              alignment_context=alignment_context,
              max_align_dist=max_align_dist,
              nearest_lev_hyperballs=nearest_lev_hyperballs,
              max_lev_dist=max_lev_dist,
              min_tok_freq = min_tok_freq,
              min_lem_freq=min_lem_freq)
        results = tag(test_items=fold_test_items,
                      WS=WS,
                      tokenize=tokenize,
                      tokenizer_context=tokenizer_context,
                      left_context=left_context,
                      right_context=right_context,
                      gazetteer=gazetteer,
                      min_tok_freq=min_tok_freq,
                      min_lem_freq=min_lem_freq,
                      mode=mode)
        # append the results of the fold:
        nr.append(results[0])
        nr_seen.append(results[1])
        nr_unseen.append(results[2])
        corr_pos_seen.append(results[3])
        corr_pos_unseen.append(results[4])
        corr_pos.append(results[5])
        corr_lem_seen.append(results[6])
        corr_lem_unseen.append(results[7])
        corr_lem.append(results[8])
        if tokenize:
            token_acc.append(results[9])
            token_f1.append(results[10])
    #########################################################################################################
    # Collect crossval-results (mean + standard deviations):
    nr = np.mean(nr)
    nr_seen_av, nr_seen_std = np.mean(nr_seen), np.std(nr_seen)
    nr_unseen_av, nr_unseen_std = np.mean(nr_unseen), np.std(nr_unseen)
    corr_pos_seen_av, corr_pos_seen_std = np.mean(corr_pos_seen), np.std(corr_pos_seen)
    corr_pos_unseen_av, corr_pos_unseen_std = np.mean(corr_pos_unseen), np.std(corr_pos_unseen)
    corr_pos_av, corr_pos_std = np.mean(corr_pos), np.std(corr_pos)
    corr_lem_seen_av, corr_lem_seen_std = np.mean(corr_lem_seen), np.std(corr_lem_seen)
    corr_lem_unseen_av, corr_lem_unseen_std = np.mean(corr_lem_unseen), np.std(corr_lem_unseen)
    corr_lem_av, corr_lem_std = np.mean(corr_lem), np.std(corr_lem)
    if tokenize:
        token_acc_av, token_acc_std = np.mean(token_acc), np.std(token_acc)
        token_f1_av, token_f1_std = np.mean(token_f1), np.std(token_f1)
    cv_info = "=================== CROSSVAL RESULTS ===================\n"
    cv_info += "Number of tokens: %s tokens\n" % (int(nr))
    cv_info += "Proportion of seen tokens: %s%% (+/- = %s)\n" % (nr_seen_av*100, nr_seen_std*100)
    cv_info += "Proportion of unseen tokens: %s%% (+/- = %s\n" % (nr_unseen_av*100, nr_unseen_std*100)
    cv_info += "Correct pos: %s%% (+/- = %s\n" % (corr_pos_av, corr_pos_std*100)
    cv_info += "\t* Correct pos seen tokens: %s%% (+/- = %s)\n" % (corr_pos_seen_av*100, corr_pos_seen_std*100)
    cv_info += "\t* Correct pos unseen tokens: %s%% (+/- = %s)\n" % (corr_pos_unseen_av*100, corr_pos_unseen_std*100)
    cv_info += "Correct lemma: %s%% (+/- = %s)\n" % (corr_lem_av*100, corr_lem_std*100)
    cv_info += "\t* Correct lemma seen tokens: %s%% (+/- = %s)\n" % (corr_lem_seen_av*100, corr_lem_seen_std*100)
    cv_info += "\t* Correct lemma unseen tokens: %s%% (+/- = %s)\n" % (corr_lem_unseen_av*100, corr_lem_unseen_std*100)
    if tokenize:
        cv_info += "\t* Tokenization scores:\n"
        cv_info += "\t* Tokenizer accuracy: %s%% (+/- = %s)\n" % (token_acc_av*100, token_acc_std*100)
        cv_info += "\t* Tokenizer f1-score: %s%% (+/- = %s)\n" % (token_f1_av*100, token_f1_std*100)
    cv_info += "========================================================"
    print(cv_info)
else:
    raise Error("Something wrong with mode parameter...")
print("::: midas ended :::")

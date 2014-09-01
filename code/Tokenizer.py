#!usr/bin/env python

import re
import os
import subprocess
import shutil

from sklearn.metrics import f1_score, accuracy_score
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier

class Tokenizer():
    def __init__(self, context=0, WS=""):
        """
        Constructor: initializes a tokenizer using sklearn's SGDClassifier
        """
        self.context = context # how many context words included on each side (symmetric right now)?
        self.WS = os.sep.join([os.path.abspath(WS), "tokenizer", ""])
        # check whether we have a working dir:
        if not os.path.isdir(self.WS):
            os.mkdir(self.WS) # create a workspace directory if it doesn't exist yet
        return

    def train(self, training_items=[], model_name="name", gazetteer=None):
        """ Function to train (and dump) a model for tokenization using sklearn's SGDClassifier """ 
        print("Training tokenizer...")
        # first collect the tokens as list:
        items = []
        for token, pos, lemma in training_items:
            # tokens that should be appended to the previous token
            # are assigned the A(ppend) category, others NA (do Not Append)
            if "~" in token:
                comps = token.lower().split("~")
                # first item of composite token doesn't have to be appended to previous (and include gazetteer info)
                items.append([comps[0], "NA"])
                # all subsequent components have to be appended to previous (and include gazetteer info)
                items.extend([[comp, "A"] for comp in comps[1:]])
            else:
                items.append([token.lower(), "NA"])
        feature_dicts = []
        train_labels = []
        for index, item in enumerate(items):
            D = {}
            focus, label = item # unpack the item
            D["focus"] = focus
            train_labels.append(label)
            if gazetteer:
                D["gazet"] = gazetteer.lookup_token(focus)
            for i in range(0, self.context):
                # right-side context tokens
                try:
                    D["t+"+str(i+1)] = items[index+i+1][0]
                except IndexError:
                    pass
                # left-side context tokens
                try:
                    D["t-"+str(i+1)] = items[index-(i+1)][0]
                except IndexError:
                    pass
            feature_dicts.append(D)
        vectorizer = DictVectorizer()
        train_X = vectorizer.fit_transform(feature_dicts)
        clf = SGDClassifier(loss="hinge", penalty="l2") # was log
        print "Training the tokenizer..."
        clf.fit(train_X, train_labels)
        # use joblib to save the model to avoid MemoryErrors:
        joblib.dump(clf, self.WS+model_name+".TOK", compress=9) # save the actual classifier
        joblib.dump(vectorizer, self.WS+model_name+".TOKenc", compress=9) # save the encoder to be able to re-encode new data
        return

    def eval_tokenizer(self, test_items=[], model_name="name", gazetteer=None):
        """ Loads pretrained tokenizer and tests it .
        Will return tuple of (accuracy, f1_score) """
        # load the vectorizer used to create the feature dict
        vectorizer = joblib.load(self.WS+model_name+".TOKenc")
        # load the actual model:
        clf = joblib.load(self.WS+model_name+".TOK")
        # extract gold tokens as they would have looked before tokenization (cf. "~" in tokens):
        gold_items = []
        for token in test_items:
            # tokens that should be appended to the previous token
            # are assigned the A(ppend) category, others NA (do Not Append)
            if "~" in token:
                comps = token.lower().split("~")
                # first item of composite token doesn't have to be appended to previous (and include gazetteer info)
                gold_items.append([comps[0], "NA"])
                # all subsequent components have to be appended to previous (and include gazetteer info)
                gold_items.extend([[comp, "A"] for comp in comps[1:]])
            else:
                gold_items.append([token.lower(), "NA"])
        # now apply the trained tokenizer:
        silver_items = []
        for index, item in enumerate(gold_items):
            D = {}
            D["focus"] = item[0]
            if gazetteer:
                D["gazet"] = gazetteer.lookup_token(D["focus"])
            for i in range(0, self.context):
                # right-side context tokens
                try:
                    D["t+"+str(i+1)] = gold_items[index+i+1][0]
                except IndexError:
                    pass
                # left-side context tokens
                try:
                    D["t-"+str(i+1)] = gold_items[index-(i+1)][0].lower()
                except IndexError:
                    pass
            D = vectorizer.transform([D])
            silver_items.append(clf.predict(D)[0])
        # now evaluate the result:
        labels = {"NA":0, "A":1}
        y_true = [labels[item[1]] for item in gold_items]
        y_pred = [labels[silver_item] for silver_item in silver_items]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
        return (acc, f1)

    def tokenize(self, test_items=[], model_name="name", gazetteer=None):
        """
        Returns tokenized version of the test_items as a list
        """
        # load the vectorizer used to create the feature dict
        vectorizer = joblib.load(self.WS+model_name+".TOKenc")
        # load the actual model:
        clf = joblib.load(self.WS+model_name+".TOK")
        tokenized = []
        for index, token in enumerate(test_items):
            D = {}
            D["focus"] = token.lower()
            if gazetteer:
                D["gazet"] = gazetteer.lookup_token(token.lower())
            for i in range(0, self.context):
                # right-side context tokens
                try:
                    D["t+"+str(i+1)] = test_items[index+i+1].lower()
                except IndexError:
                    pass
                # left-side context tokens
                try:
                    D["t-"+str(i+1)] = test_items[index-(i+1)].lower()
                except IndexError:
                    pass
            D = vectorizer.transform([D])
            result = clf.predict(D)[0]
            if result == "NA":
                tokenized.append(token)
            else:
                try:
                    # try to append the new token to the previous one
                    tokenized[-1] = tokenized[-1]+"~"+token # : an exception to catch the rare situation in which
                                                            # the very first token would have to be appended according
                                                            # to the tokenizer. In that case we simply append it.
                except IndexError:
                    tokenized.append(token)
        return tokenized


        
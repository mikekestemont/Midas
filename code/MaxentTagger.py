#!usr/bin/env python

import os
import re
import cPickle as pickle
import maxent

from Lemmatizer import *

def get_prefix_suffix(w, length):
    # Taken from https://github.com/lzhang10/maxent/blob/master/example/postagger/postagger.py [25 August 2014]
    # Copyright (C) 2003 by Zhang Le <ejoy@users.sourceforge.net>
    l = min(len(w),length)+1
    p = []
    s = []
    for i in range(l):
        p.append("pf_"+w[:i+1])
        s.append("sf_"+w[-i:])
    return p, s

class MaxentTagger(object):
    def __init__(self, WS="", left_context=0, right_context=0, min_tok_freq=0, min_lem_freq=0):
        # dir under global workspace where the tagger can create a dir to dump intermediary files:
        self.left_context = left_context
        self.right_context = right_context
        self.min_tok_freq = min_tok_freq # only tokens with a freq above this threshold will be considered for the representation of the context
        self.min_lem_freq = min_lem_freq # only lemmas with a freq above this threshold will be considered for the representation of the context
        # check whether we have a working dir:
        self.WS = os.sep.join([os.path.abspath(WS), "postagger", ""])
        if not os.path.isdir(self.WS):
            os.mkdir(self.WS) # create a workspace directory if it doesn't exist yet
        return

    def train(self, training_items=[], model_name="name", gazetteer=None):
        """
        Wrapper for training a sequential tagger on the training data using MaxEnt
        """
        # First, determine frequency thresholds
        self.token_freqs = {}
        self.lemma_freqs = {}
        print "\t* Extracting token and lemma frequencies..."
        for token, _, lemma in training_items:
            if token == "<utt>":
                continue
            else:
                token = token.replace("~","").lower()
                try:
                    self.token_freqs[token] += 1
                except KeyError:
                    self.token_freqs[token] = 1
                try:
                    self.lemma_freqs[lemma] += 1
                except KeyError:
                    self.lemma_freqs[lemma] = 1
        # Initialize the maxent tagger:
        maxent.set_verbose(1)
        # two separate models for pos-tagging & lemmatization:
        m_pos, m_lem = maxent.MaxentModel(), maxent.MaxentModel()
        m_pos.begin_add_event()
        m_lem.begin_add_event()
        for index, train_item in enumerate(training_items):
            focus_token, focus_pos, focus_lemma = train_item
            if focus_token == "<utt>":
                continue
            pos_context, lem_context = [], []            
            if focus_token[0].isupper():
                pos_context.append("t=upp")
                lem_context.append("t=upp")
            focus_token = focus_token.replace("~", "").lower()
            # add the (lowercased) focus token:
            pos_context.append("t="+focus_token)
            lem_context.append("t="+focus_token)
            # add the gazetteer info for the focus token:
            if gazetteer:
                g = gazetteer.lookup_token(focus_token)
                pos_context.append("g="+g)
                lem_context.append("g="+g)
            # get the target outcome:
            pos_outcome = focus_pos
            lem_outcome = focus_lemma
            # add to pos outcome to the lem_model:
            lem_context.append("p="+pos_outcome)
            # now walk through context:
            # first: right context (no lemmas or pos available because of left-to-right strategy!)
            for i in xrange(0, self.right_context):
                # token:
                try:
                    token = training_items[index+i+1][0].replace("~","").lower()
                    if token == "<utt>" or self.token_freqs[token] >= self.min_tok_freq:
                        pos_context.append("t+"+str(i+1)+"="+token)
                        lem_context.append("t+"+str(i+1)+"="+token)
                except (IndexError, KeyError):
                     pass
            # now left_context (tokens, pos and lemmas)
            for i in xrange(0, self.left_context):
                # token:
                try:
                    token = training_items[index-(i+1)][0].replace("~","").lower()
                    if token == "<utt>" or self.token_freqs[token] >= self.min_tok_freq:
                        pos_context.append("t-"+str(i+1)+"="+token)
                        lem_context.append("t-"+str(i+1)+"="+token)
                except (IndexError, KeyError):
                     pass
                # pos:
                try:
                    pos = training_items[index-(i+1)][1]
                    pos_context.append("p-"+str(i+1)+"="+pos)
                    lem_context.append("p-"+str(i+1)+"="+pos)
                except IndexError:
                     pass
                # lemma:
                try:
                    lemma = training_items[index-(i+1)][2]
                    if lemma == "<utt>" or self.lemma_freqs[lemma] >= self.min_lem_freq:
                        pos_context.append("l-"+str(i+1)+"="+lemma)
                        lem_context.append("l-"+str(i+1)+"="+lemma)
                except (IndexError, KeyError):
                     pass
            for affixes in get_prefix_suffix(focus_token, 3):
                pos_context.extend(affixes)
                lem_context.extend(affixes)
                # we currently ignore lemmas that have more than three components:
            if focus_lemma.count("+") <= 3:
                m_pos.add_event(pos_context, pos_outcome)
                if focus_token == "<utt>" or self.lemma_freqs[focus_lemma] >= self.min_lem_freq:
                    m_lem.add_event(lem_context, lem_outcome)
        m_pos.end_add_event()
        m_lem.end_add_event()
        print "Training postagger..."
        m_pos.train(150, "lbfgs")
        m_pos.save(self.WS+model_name+".POS", True)
        print "Training lemmatizer..."
        m_lem.train(75, "lbfgs")
        m_lem.save(self.WS+model_name+".LEM", True)
        return

    def train_lemmatizer(self, training_items=[], model_name="name", alignment_context=1,
                         max_align_dist=2, nearest_lev_hyperballs=2, max_lev_dist=3):
        """ Train specific functionality for the lemmatizer """
        self.lemmatizer = Lemmatizer()
        self.lemmatizer.extract_lexica(training_items=training_items) # extract useful lexica from the training data
        self.lemmatizer.train_alignator(alignment_context=alignment_context,
                                        max_align_dist=max_align_dist)
        self.lemmatizer.get_levenshtein_suggester(nearest_lev_hyperballs=nearest_lev_hyperballs,
                                                  max_lev_dist=max_lev_dist)
        # pickle some of training-related objects for later use:
        pickle.dump(self.lemmatizer, open(self.WS+"lemmatizer_"+model_name+".p", "wb"))
        pickle.dump(self.token_freqs, open(self.WS+"token_freqs"+model_name+".p", "wb"))
        pickle.dump(self.lemma_freqs, open(self.WS+"lemma_freqs"+model_name+".p", "wb"))
        return

    def load_models(self, model_name="name"):
        """Function that loads previously trained models saved under the workspace dir."""
        maxent.set_verbose(1)
        self.M_pos = maxent.MaxentModel()
        self.M_pos.load(self.WS+model_name+".POS")
        self.M_lem = maxent.MaxentModel()
        self.M_lem.load(self.WS+model_name+".LEM")
        self.lemmatizer = pickle.load(open(self.WS+"lemmatizer_"+model_name+".p", "rb"))
        self.token_freqs = pickle.load(open(self.WS+"token_freqs"+model_name+".p", "rb"))
        self.lemma_freqs = pickle.load(open(self.WS+"lemma_freqs"+model_name+".p", "rb"))
        return

    def tag(self, tokenized=[], gazetteer=None):
        # declare a container for the silver predictions:
        predicted_pos, predicted_lemmas = [], []
        # loop over test tokens:
        for index, token in enumerate(tokenized):
            if token == "<utt>":
                predicted_pos.append("<utt>")
                predicted_lemmas.append("<utt>")
                continue
            pos_context, lem_context = [], []
            focus_token = tokenized[index]
            if focus_token[0].isupper():
                pos_context.append("t=upp")
                lem_context.append("t=upp")
            focus_token = focus_token.replace("~", "").lower()
            pos_context.append("t="+focus_token)
            if gazetteer:
                pos_context.append("g="+gazetteer.lookup_token(focus_token))
            # first: right context (no lemmas: will not be known for right context at tagging time because of left2right strategy)
            for i in xrange(0, self.right_context):
                # token:
                try:
                    token = tokenized[index+i+1].replace("~","").lower()
                    if token == "<utt>" or self.token_freqs[token] >= self.min_tok_freq:
                        pos_context.append("t+"+str(i+1)+"="+token)
                except (IndexError, KeyError):
                     pass
            # now left_context (tokens, pos and lemmas)
            for i in xrange(0, self.left_context):
                # token:
                try:
                    token = tokenized[index-(i+1)].replace("~","").lower()
                    if token == "<utt>" or self.token_freqs[token] >= self.min_tok_freq:
                        pos_context.append("t-"+str(i+1)+"="+token)
                except (IndexError, KeyError):
                     pass
                # pos:
                try:
                    pos = predicted_pos[index-(i+1)]
                    pos_context.append("p-"+str(i+1)+"="+pos)
                except IndexError:
                     pass
                # lemma:
                try:
                    lemma = predicted_lemmas[index-(i+1)]
                    if lemma == "<utt>" or self.lemma_freqs[lemma] >= self.min_lem_freq:
                        lem_context.append("l-"+str(i+1)+"="+lemma)
                except (IndexError, KeyError):
                    pass
            for affixes in get_prefix_suffix(focus_token, 3):
                pos_context.extend(affixes)
            pos_options = self.M_pos.eval_all(pos_context)
            selected_pos = pos_options[0][0]
            lem_context+=pos_context
            if focus_token in self.lemmatizer.token2lemmas:
    #           print focus_token+" (seen)"
                silver_lemma, silver_pos = "<unk>", "<unk>"
                # try to get an unambiguous result given the test_token and the pos tag proposed by the tagger:
                unambiguous_result = self.lemmatizer.lemmatize_seen_token(focus_token, selected_pos)
                if unambiguous_result:
                    silver_lemma, silver_pos = unambiguous_result
                else:
                    silver_pos = selected_pos
                    potential_lemmas = self.lemmatizer.token2lemmas[focus_token]
                    lem_context.append("p="+silver_pos)
                    for item in self.M_lem.eval_all(lem_context):
                        if item[0] in potential_lemmas:
                            silver_lemma = item[0]
                            break
    #           print "\t - suggested lemma: "+str(silver_lemma)
    #           print "\t - suggested pos: "+str(silver_pos)
            else:
    #           print focus_token+" (unseen, known token)"
                # lemmatize the unseen token
                tmp_results = self.lemmatizer.lemmatize_unseen_token(focus_token, selected_pos, pos_options, self.lemma_freqs)
                # unpack:
                lev_majority_lemma, suggested_lev_lemmas, reranked_lemma, silver_lemma, silver_pos = tmp_results
                # update the statistics:
    #           print "\t - suggested lemma: "+str(silver_lemma)
    #           print "\t - suggested pos: "+str(silver_pos)
            predicted_pos.append(silver_pos)
            predicted_lemmas.append(silver_lemma)
        return zip(tokenized, predicted_pos, predicted_lemmas)

#!usr/bin/env python

from operator import itemgetter
import os

from Alignator import *
from Levenshtein_suggester import *

class Lemmatizer(object):
    """
    Module to lemmatize words. Uses the pos distribution predicted by the pos tagger.
    """
    def __init__(self):
        """
        Simple constructor
        """
        # declare empty dicts
        self.token2lemmas = {}
        self.lemma2tokens = {}
        self.lemma2pos = {}
        return

    def extract_lexica(self, training_items=[]):
        """
        Extract useful lexica from training data
        """
        for token, pos, lemma in training_items:
            token = token.replace("~", "")
            try:
                self.token2lemmas[token].add(lemma)
            except KeyError:
                self.token2lemmas[token] = set()
                self.token2lemmas[token].add(lemma)
            try:
                self.lemma2tokens[lemma].add(token)
            except KeyError:
                self.lemma2tokens[lemma] = set()
                self.lemma2tokens[lemma].add(token)
            try:
                self.lemma2pos[lemma].add(pos)
            except KeyError:
                self.lemma2pos[lemma] = set()
                self.lemma2pos[lemma].add(pos)
        return

    def train_alignator(self, alignment_context=2, max_align_dist=3):
        """
        Trains a alignator that can estimate the global likelihood of a alignation between two tokens.
        Outsources this to an Alignator object
        """
        self.alignator = Alignator(context=alignment_context,
                                   max_align_dist=max_align_dist)
        self.alignator.train(lemma2tokens=self.lemma2tokens)
        return

    def get_levenshtein_suggester(self, nearest_lev_hyperballs=2, max_lev_dist=3):
        """
        Fits a a Levenshtein module which returns the training tokens,
        which are closest in plain edit dist to an unseen token.
        """
        self.levenshtein_suggester = Levenshtein_suggester(nearest_lev_hyperballs=nearest_lev_hyperballs,
                                                           max_lev_dist=max_lev_dist)
        return

    def lemmatize_seen_token(self, token, selected_pos):
        # is their just one option, given the pos tag selected?
        candidates = set([lemma for lemma in self.token2lemmas[token] if selected_pos in self.lemma2pos[lemma]])
        if len(candidates) == 1:
            return (list(candidates)[0], selected_pos)
        else:
            return None # in case of results that cannot be disambiguated on the basis of the pos tag predicted

    def lemmatize_unseen_token(self, token, selected_pos, pos_options, lemma_freqs):
        # function returns a list: [pure Levenshtein lemma, lemma after reranking through alignment,
        #                               lemma after contextual disambiguation, disambiguated pos tag]
        pos_options = dict((k, s) for k, s in pos_options)
        token = token.replace("~","")
        levenshtein_candidates = self.levenshtein_suggester.get_levenshtein_candidates(token, self.token2lemmas)
        if not levenshtein_candidates:
            return ["<unk>", [], "<unk>", "<unk>", selected_pos]
        levenshtein_candidates, majority_lemma_closest_lev_token, lev_suggested_lemmas =\
                self.levenshtein_suggester.sort_candidates(levenshtein_candidates, lemma_freqs)
        hyperballs = self.levenshtein_suggester.candidates2hyperballs(levenshtein_candidates)
        reranking, random_lemma_reranked_token = self.alignator.rerank(token, hyperballs)
        if len(reranking) == 1:
            return [majority_lemma_closest_lev_token, lev_suggested_lemmas, majority_lemma_closest_lev_token, random_lemma_reranked_token, selected_pos]
        contextual_ranking = []
        for token, lemma, lexical_score in reranking:
#           print("lemma: "+lemma)
#           print(token, lemma, lexical_score)
            for pos in self.lemma2pos[lemma]:
                if pos in pos_options:
                    contextual_score = pos_options[pos]
                    if contextual_score < 0.001:
                        continue
                    contextual_ranking.append([lemma, pos, lexical_score*contextual_score])
#                   print('\t\tlexical score: '+str(lexical_score))
#                   print('\t\tpos score: '+str(contextual_score))
#                   print('\t\tcombined score: '+str(contextual_score*contextual_score))
        contextual_ranking = sorted(contextual_ranking, key=itemgetter(2), reverse=True)
        if len(contextual_ranking) > 0:
            disambiguated_lemma = contextual_ranking[0][0]
            disambiguated_pos = contextual_ranking[0][1]
        else:
            disambiguated_lemma = reranking[0][1]
            disambiguated_pos = selected_pos
        return [majority_lemma_closest_lev_token, lev_suggested_lemmas, random_lemma_reranked_token, disambiguated_lemma, disambiguated_pos]

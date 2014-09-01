#!usr/bin/env python

from operator import itemgetter

import editdist

class Levenshtein_suggester(object):
    """
    Class that returns the training tokens closest to (unseen) tokens, together with their lemmas.
    """
    def __init__(self, nearest_lev_hyperballs=2, max_lev_dist=3):
        """
        Simple constructor
        """
        self.nearest_lev_hyperballs = nearest_lev_hyperballs
        self.max_lev_dist = max_lev_dist
        return

    def get_levenshtein_candidates(self, test_token="token", token2lemmas={}):
        """
        Function returns an initial, rough selection of levenshtein candidates
        """
        candidates = []
        for train_token in token2lemmas:
            train_lemmas = token2lemmas[train_token]
            # calculate the edit distance between the test_token and all seen tokens
            edit_dist =  editdist.distance(test_token, train_token)
            # append the training item as a candidate if it is close enough:
            if edit_dist <= self.max_lev_dist:
                candidates.append([train_token, train_lemmas, edit_dist])
        if candidates:
            return candidates

    def sort_candidates(self, candidates, lemma_freqs):
        """
        Returns a list: [candidates sorted on the basis of their the edit dist,
                                majority lemma for the closest item (for evaluation purposes), 
                                list of suggested lemmas]
        """
        freq_ranking = [] # a ranking of the suggested lemma based on their training freq
        candidates = sorted(candidates, key=itemgetter(2), reverse=True)
        lev_suggested_lemmas = set()
        for train_token, train_lemmas, dist in candidates:
            for lemma in train_lemmas:
                lev_suggested_lemmas.add(lemma)
                try:
                    freq_ranking.append((lemma, lemma_freqs[lemma]))
                except KeyError:
                    pass
        freq_ranking = sorted(freq_ranking, key=itemgetter(1), reverse=True)
        majority_lemma_closest_lev_token = freq_ranking[0][0]
        return (candidates, majority_lemma_closest_lev_token, lev_suggested_lemmas)

    def candidates2hyperballs(self, candidates):
        # function that distributes the closest tokens over hyperballs
        hyperballs = []
        # declare tmp variable:
        hyperball = []
        current_distance = 0
        for candidate in candidates:
            candidate_distance = candidate[2]
            # check whether the current distance isn't too large already:
            if candidate_distance > self.max_lev_dist:
                if len(hyperball) > 0: # don't forget to flush the previous hyperball before we leave the loop
                    hyperballs.append(hyperball)
                break
            else:
                if candidate_distance == current_distance: # if same distance as previous candidate
                    for lemma in candidate[1]: # extend the existing hyperball
                        hyperball.append([candidate[0],lemma])
                elif candidate_distance > current_distance: # larger distance than previous
                    if len(hyperball) > 0: # if there already existed a hyperball, we add the item
                        hyperballs.append(hyperball)
                    if candidate_distance > self.max_lev_dist:
                        break
                    else:
                        hyperball = [] # ... else, we create a hyperball and add the new item
                        for lemma in candidate[1]:
                            hyperball.append([candidate[0],lemma]) # ... which we fill with the current candidate
                        current_distance = candidate_distance # we update the distance
        # don't forget to flush the last one:
        if len(hyperball) > 0:
            hyperballs.append(hyperball)
        return hyperballs
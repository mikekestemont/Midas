#!usr/bin/env python

import editdist
from operator import itemgetter

from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier

from Transliterator import *

class Alignator(object):
    """
    Class that will align tokens and estimate the global probability of an alignation.
    """
    def __init__(self, context=2, max_align_dist=3):
        """
        Constructor
        """
        self.transliterator = Transliterator(context) # outsources the actual alignment to a submodule
        self.max_align_dist = max_align_dist
        return

    def train(self, lemma2tokens={}):
        """
        Train the align by pairing each pair of tokens under a given lemma,
        which have an edit dist to each other < maximum alignment distance
        (i.e. also identical tokens are aligned to not over-abstract)
        """
        # declare containers for the alignment vectors:
        self.train_feature_dicts = []
        self.train_labels = []
        for lemma in lemma2tokens:
            tokens = lemma2tokens[lemma]
            for t1 in tokens:
                for t2 in tokens:
                    # combine and align items inside lemma that are close enough (or identical!):
                    if (t1 == t2) or (editdist.distance(t1, t2) < self.max_align_dist):
                        curr_dicts, curr_labels = self.transliterator.transliterate(t1, t2)
                        self.train_feature_dicts.extend(curr_dicts)
                        self.train_labels.extend(curr_labels)
        # transform the alignment vectors dicts into sklearn format
        self.vectorizer = DictVectorizer()
        self.train_X = self.vectorizer.fit_transform(self.train_feature_dicts)
        self.labelEncoder = preprocessing.LabelEncoder()
        self.train_y = self.labelEncoder.fit_transform(self.train_labels)
        self.clf = SGDClassifier(loss="log", penalty="l2")
        print("Training the SGD classifier for the alignator...")
        self.clf.fit(self.train_X, self.train_y)
        return

    def rerank(self, token, hyperballs):
        """ Functions that reranks the tokens in the hyperballs passed,
        by aligning it with each candidate and estimating the global 
        probability of the transliteration.
        """
        # unroll the candidates from the hyperballs to a 2D list:
        candidates = []
        for hyperball in hyperballs:
            for tok, lem in hyperball:
                candidates.append([tok, lem])
        alignment_ranking = []
        for candidate_token, candidate_lemma in candidates:
#           print("\t* "+str(candidate_token))
            # now, we align each candidate with the test_token
            test_dicts, test_labels = self.transliterator.transliterate(t1=token, t2=candidate_token)
            # check we are not trying to predict entirely new transformation:
            for index, test_label in enumerate(test_labels):
                # if we have to predict an unknown test_label, we backoff to the common char "e"
                if test_label not in self.labelEncoder.classes_:
                    test_labels[index] = "e"
            test_X = self.vectorizer.transform(test_dicts)
            test_y = self.labelEncoder.transform(test_labels)
#            for d, l in zip(test_dicts,test_labels):
#                print("\t\t+ "+str(d), end="")
#                print(" = "+str(l))
            results = self.clf.predict_proba(test_X)
            # initialize the total score to 1 for the iterative product:
            total_score = 1
            # loop over the alignments:
            for label, result, test_dict in zip(test_y, results, test_dicts):
#               print(test_dict["focus"]+"============")
#               print(alignment_labelEncoder.inverse_transform(label)+"************")
                score_target_label = result[label]
#               print "\ty"+str(score_target_label)+"%%%%%%%%%%%%%%%"
                total_score *= score_target_label # multiply the probabilities
            alignment_ranking.append([candidate_token, candidate_lemma, total_score])
        # now rank the candidates for their global score:
        alignment_ranking = sorted(alignment_ranking, key=itemgetter(2), reverse=True)
        # we now have a reranking list of candidates:
#       print("Alignment ranking for "+token+": "+str(alignment_ranking))
        # select the single best and return it with the list
        lemma_closest_reranked_token = alignment_ranking[0][1]
        return (alignment_ranking, lemma_closest_reranked_token)


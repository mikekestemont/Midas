#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script by Grzegorz Chrupała: miscelaneous functions analyzing tagged and lemmatized data
"""

import argparse
from __future__ import division
import sys
from operator import itemgetter

def parse(inp):
    '''Yields stream of documents, where each document is a list of
    paragraphs, and each paragraph consists of a list of triples
    (word, tag, lemma).'''
    doc = []
    par = []
    for line in inp:
        #print par
        if is_docsep(line):
            doc.append(par)
            par = []
            yield [ x for x in doc ]
            doc = []
        elif is_linebreak(line):
            doc.append([ x for x in par])
            par = []
        else:
            token = parse_line(line)
            par.append(token)
    if len(par) > 0:
        doc.append(par)
    if len(doc) > 0:
        yield doc

def is_docsep(x):
    if len(x.strip()) == 0:
        return False
    f1 = x.split("\t")
    return x[0] == '@'

def is_linebreak(x):
    return len(x.strip()) == 0

def parse_line(x):
    "Parses line into a (word, pos, lemma) triple."
    fs = x.strip().split()
    if len(fs) != 3:
        raise ValueError("wrong data {0}".format(repr(x)))
    else: 
        word, pos, lemma = fs
        return (word, pos, lemma)

def tokens(src):
    "Yields tokens from paragraphs in a sequence of documents."
    for doc in src:
        for par in doc:
            for tok in par:
                yield tok


# Evaluation

def report(goldf, predf, fields, train_file=None):
    if train_file:
        train = list(tokens(parse(open(train_file))))
    else:
        train = []
    
    seen_form = set([ tok[0] for tok in train])
    seen_full = set([ tok for tok in train])

    predtoks = tokens(parse(open(predf)))
    goldtoks = tokens(parse(open(goldf)))
    
    get = itemgetter(*fields)
    
    gold_pred_toks      = zip(goldtoks, predtoks)
    gold_pred_all       =  [ (get(g), get(p)) for (g, p) in gold_pred_toks ]
    gold_pred_seen_form =  [ (get(g), get(p)) for (g, p) in gold_pred_toks 
                             if g[0] in seen_form and g not in seen_full ]
    gold_pred_seen_full =  [ (get(g), get(p)) for (g, p) in gold_pred_toks if g in seen_full ]
    gold_pred_unseen    =  [ (get(g), get(p)) for (g, p) in gold_pred_toks if g[0] not in seen_form]
    
    return { "all":         error_rate(gold_pred_all),
             "seen_form":   error_rate(gold_pred_seen_form), 
             "seen_full":   error_rate(gold_pred_seen_full),
             "unseen":      error_rate(gold_pred_unseen)
           }
             
    
    
def error_rate(gold_pred):
   errno = sum((1 if g != p else 0 for (g,p) in gold_pred))
   size = len(gold_pred)
   if size == 0.0: 
       return (0.0, 0.0, None)
   else:
       return (errno, size, round(errno / size, 3))
              
def paras(src):
    "Splits src into paragraphs separated by newlines"
    current = []
    for line in src:
        if line.strip() == "":
            yield current
            current = []
        else: 
            current.append(line)
    if current:
        yield current


def main():
    parser = argparse.ArgumentParser(description="Evaluate tagging and lemmatization")
    parser.add_argument("gold_file", help="path to file with gold labels")
    parser.add_argument("pred_file", help="path to file with predicted labels")
    parser.add_argument("--train_file", help="path to file with training data")
    
    args = parser.parse_args()

    results = {"pos"   : report(args.gold_file, args.pred_file, [1],   train_file=args.train_file),
               "lemma" : report(args.gold_file, args.pred_file, [2],   train_file=args.train_file),
               "both"  : report(args.gold_file, args.pred_file, [1,2], train_file=args.train_file),
              }
    print "error\tno\tsize\ttype\tsubset"
    for label_type in results:
        for subset in ["all","seen_form","seen_full","unseen"]:
            score = results[label_type][subset]
            print "{0}\t{1}\t{2}\t{3}\t{4}".format(score[2],
                                                   score[0],
                                                   score[1], 
                                                   label_type, 
                                                   subset)


            
if __name__ == '__main__':
    main()



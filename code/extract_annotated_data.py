#!usr/bin/env python

"""
Standalone, auxiliary script to extract the original, annotated data to:
(a) get splits to train and test our system for the Clips paper
(b) get the entire normalized version of each corpus (to train Midas)
"""

import re, os, codecs
from operator import itemgetter

from TagConverter import *

SEPARATOR = re.compile("(<C )|(<q>)")
ABBR_TAGS = re.compile(r"(<A >)|(</A>)")
VN_TYPE = re.compile(r"<VN type=[0-9]+>")
WHITESPACE = re.compile(r"\s+")
NUMBERS = re.compile(r'[0-9]+')
TAB = re.compile(r'\t')
PUNCTUATION = re.compile(r"\(|\)|\"|\(|\)|\=|\)|\[|\]|\<|\>|\'|\{|\}|\\|\_|\@|")

def clean_token(dirtyT):
    cleanT = PUNCTUATION.sub("", dirtyT)
    cleanT = WHITESPACE.sub("", cleanT)
    return cleanT

def cg_lit():
    # set paths:
    orig_dir = "../data/original/annotated/cg-lit/"
    fulltextD = "../data/uniform/annotated/cg-lit/full/"
    trainFname = "../data/uniform/annotated/cg-lit/train/cg-lit_train.3col"
    devFname = "../data/uniform/annotated/cg-lit/dev/cg-lit_dev.3col"
    testFname = "../data/uniform/annotated/cg-lit/test/cg-lit_test.3col"
    # create file to append parsed items to:
    with open(trainFname, 'w+') as trainF:
        trainF.write("\n")
    trainF = open(trainFname, 'a+')
    with open(devFname, 'w+') as devF:
        devF.write("\n")
    devF = open(devFname, 'a+')
    with open(testFname, 'w+') as testF:
        testF.write("\n")
    testF = open(testFname, 'a+')
    # parse the original files:
    for textN in os.listdir(orig_dir):
        if textN.endswith(".fromdb"):
            print textN
            all_lines = []
            oldF = open(orig_dir+textN, 'r')
            for line in [line.strip() for line in oldF.readlines()]:
                if line.startswith("<L "): # we are only interested in text lines
                    current_line = []
                    line = VN_TYPE.sub(" ", line) # remove <VN type=1> tags etc.
                    line = ABBR_TAGS.sub(" ", line) # remove abbreviation indications
                    for comp in SEPARATOR.split(line):
                        if comp:
                            comp = comp.strip()
                            token, pos, lemma = "", "", ""
                            if not comp.startswith("<L ") or not comp[0].isdigit():
                                try:
                                    rest, token = comp.split("> ")
                                    pos, lemma = rest.split("_")
                                    token = clean_token(token)
                                    token = token.replace("+", "~")
                                    if "..." in token:
                                        continue
                                    pos = pos.replace("#", "").replace("*", "") # remove indications we don't use...
                                    lemma = lemma.replace("-", "").strip() # remove Prtcl-indications
                                    if "+" in pos and "+" in lemma:
                                        pos_comps = pos.split("+")
                                        lemma_comps = lemma.split("+")
                                        if len(pos_comps) == len(lemma_comps):
                                            for c in range(len(pos_comps)):
                                                while len(pos_comps[c]) < 3:
                                                    pos_comps[c] = "0"+pos_comps[c]
                                                pos_comps[c] = tag_converter.Num2Ad(pos_comps[c])
                                                pos_comps[c] = tag_converter.removeAdEnding(pos_comps[c])
                                            annotations = zip(lemma_comps, pos_comps)
                                            annotations.sort(key=itemgetter(0))
                                            pos = "+".join([item[1] for item in annotations])
                                            lemma = "+".join([item[0] for item in annotations])
                                    else:
                                        while len(pos) < 3:
                                            pos = "0"+pos
                                            if lemma == "frans" and pos == "999":
                                                lemma = token
                                        pos = tag_converter.Num2Ad(pos)
                                        pos = tag_converter.removeAdEnding(pos)
                                except ValueError:
                                    pass
                                # remove "LATIJN"
                                lemma = lemma.lower()
                                if lemma == "latijn" or lemma == "zzz":
                                    lemma = token
                                item = (token.lower()+"\t"+pos+"\t"+lemma.strip().lower()).strip()
                                if item:
                                    if not pos.strip() or not lemma.strip() or not token.strip():
                                        print token
                                        print pos
                                        print lemma
                                        print line
                                    current_line.append(item)
                    if current_line: # flush previous line
                        all_lines.append(current_line)
            oldF.close()
            # first full text files:
            fulltextF = codecs.open(fulltextD+textN.replace(".fromdb", ".3col"), "w+", "utf-8")
            fulltextF.write("\n")
            for line in all_lines:
                for item in line:
                    try:
                        fulltextF.write(item+"\n")
                    except UnicodeDecodeError:
                        pass
                fulltextF.write("\n")
            fulltextF.close()
            # now split per text:
            fold_length = int(len(all_lines)/10)
            # train (first part):
            trainF.write("@ begin_of_"+textN+"\n")
            for i in range(0, fold_length*6):
                line = all_lines[i]
                for item in line:
                    try:
                        trainF.write(item+"\n")
                    except UnicodeDecodeError:
                        pass
                trainF.write("\n")
            # train (second part, from after dev and test):
            for i in range(fold_length*8, fold_length*10):
                line = all_lines[i]
                for item in line:
                    try:
                        trainF.write(item+"\n")
                    except UnicodeDecodeError:
                        pass
                trainF.write("\n")
            # dev:
            devF.write("@ begin_of_"+textN+"\n")
            for i in range(fold_length*6, fold_length*7):
                line = all_lines[i]
                for item in line:
                    try:
                        devF.write(item+"\n")
                    except UnicodeDecodeError:
                        pass
                devF.write("\n")
            # test:
            testF.write("@ begin_of_"+textN+"\n")
            for i in range(fold_length*7, fold_length*8):
                line = all_lines[i]
                for item in line:
                    try:
                        testF.write(item+"\n")
                    except UnicodeDecodeError:
                        pass
                testF.write("\n")
    trainF.close()
    devF.close()
    testF.close()
    return

def relig():
    print 'relig...'
    orig_dir = "../data/original/annotated/relig/"
    fulltextD = "../data/uniform/annotated/relig/full/"
    trainFname = "../data/uniform/annotated/relig/train/relig_train.3col"
    devFname = "../data/uniform/annotated/relig/dev/relig_dev.3col"
    testFname = "../data/uniform/annotated/relig/test/test_relig_test.3col"
    with open(trainFname, 'w+') as trainF:
        trainF.write("\n")
    trainF = open(trainFname, 'a+')
    with open(devFname, 'w+') as devF:
        devF.write("\n")
    devF = open(devFname, 'a+')
    with open(testFname, 'w+') as testF:
        testF.write("\n")
    testF = open(testFname, 'a+')
    for textN in os.listdir(orig_dir):
        if textN.endswith(".txt"):
            print textN
            all_items = []
            oldF = open(orig_dir+textN, 'r')
            for line in [line.strip() for line in oldF.readlines() if line.strip()]:
                try:
                    token, pos, lemma = TAB.split(line)
                except ValueError:
                    print line
                    raise ValueError
                token = clean_token(token)
                lemma = lemma.lower()
                if lemma == "latijn" or lemma == "zzz":
                    lemma = token
                item = (token.lower()+"\t"+pos+"\t"+lemma.strip()).strip()
                if item:
                    if not pos.strip() or not lemma.strip() or not token.strip():
                        print line
                    all_items.append(item)
            oldF.close()
            # first full text files:
            fulltextF = codecs.open(fulltextD+textN, "w+", "utf-8")
            fulltextF.write("\n")
            for item in all_items:
                try:
                    fulltextF.write(item+"\n")
                except UnicodeDecodeError:
                    pass
            fulltextF.close()
            # now split per text:
            fold_length = int(len(all_items)/10)
            # train (first slice):
            trainF.write("@ begin_of_"+textN+"\n")
            for i in range(0, fold_length*6):
                try:
                    trainF.write(all_items[i]+"\n")
                except UnicodeDecodeError:
                    pass
            # train (second slice):
            for i in range(fold_length*8, fold_length*10):
                try:
                    trainF.write(all_items[i]+"\n")
                except UnicodeDecodeError:
                    pass
            # dev:
            devF.write("@ begin_of_"+textN+"\n")
            for i in range(fold_length*6, fold_length*7):
                try:
                    devF.write(all_items[i]+"\n")
                except UnicodeDecodeError:
                    pass
            testF.write("@ begin_of_"+textN+"\n")
            # test:
            for i in range(fold_length*7, fold_length*8):
                try:
                    testF.write(all_items[i]+"\n")
                except UnicodeDecodeError:
                    pass
    trainF.close()
    devF.close()
    testF.close()
    return


def cgadmin():
    # first rank the charters chronologically:
    SPLITTER = re.compile(r'p|r|a|c')
    meta = []
    with open("../data/original/annotated/cg-admin.txt", 'r') as oldF:
        for line in [line.strip() for line in oldF.readlines() if line.strip()]:
            if line.startswith("@"):
                name = line.split()[3]
                year = int("1"+SPLITTER.split(name)[1][:3])
                meta.append([name,year])
    meta.sort(key=itemgetter(1), reverse=False)
    lookup = {}
    for i in range(len(meta)):
        if str(i)[-1] == "9":
            lookup[meta[i][0]] = "DEV"
        elif str(i)[-1] == "0":
            lookup[meta[i][0]] = "TEST"
        else:
            lookup[meta[i][0]] = "TRAIN"
    with open("../data/uniform/annotated/cg-admin/full/cg-admin.3col", "w+") as fullF:
        fullF.write("\n")
    fullF = open("../data/uniform/annotated/cg-admin/full/cg-admin.3col", "a+")
    with open("../data/uniform/annotated/cg-admin/train/cg-admin_train.3col", "w+") as trainF:
        trainF.write("\n")
    trainF = open("../data/uniform/annotated/cg-admin/train/cg-admin_train.3col", "a+")
    with open("../data/uniform/annotated/cg-admin/dev/cg-admin_dev.3col", "w+") as devF:
        devF.write("\n")
    devF = open("../data/uniform/annotated/cg-admin/dev/cg-admin_dev.3col", "a+")
    with open("../data/uniform/annotated/cg-admin/test/cg-admin_test.3col", "w+") as testF:
        testF.write("\n")
    testF = open("../data/uniform/annotated/cg-admin/test/cg-admin_test.3col", "a+")
    F = None
    with open("../data/original/annotated/cg-admin.txt", 'r') as oldF:
        for line in [line.strip() for line in oldF.readlines() if line.strip()]:
            if line.startswith("@"):
                # determine file to write to
                name = line.split()[3]
                if lookup[name] == "TRAIN":
                    F = trainF
                elif lookup[name] == "DEV":
                    F = devF
                elif lookup[name] == "TEST":
                    F = testF
                else:
                    print "Ooooops..."
                F.write("@ begin_of_"+name+"\n")
                fullF.write("@ begin_of_"+name+"\n")
            else:
                try:
                    comps = WHITESPACE.split(line)
                    token, lemma, pos = comps[2], comps[-2], comps[-1]
                    token = clean_token(token)
                    token = token.replace("+", "~")
                    if len(token)<1 or len(lemma)<1 or len(pos)<3 or "..." in token:
                        continue
                    pos = pos.replace("#", "").replace("*", "").replace("_", "") # remove indications we don't use...
                    lemma = lemma.replace("-", "").lower()
                    if lemma == "frans" and pos == "999":
                        lemma = token
                    if "+" in pos:
                        pos_comps = pos.split("+")
                        for i in range(len(pos_comps)):
                            pos_comps[i] = tag_converter.Num2Ad(pos_comps[i])
                            pos_comps[i] = tag_converter.removeAdEnding(pos_comps[i])
                        pos = "+".join(pos_comps)
                    else:
                        pos = tag_converter.Num2Ad(pos)
                        pos = tag_converter.removeAdEnding(pos)
                    if lemma == "latijn" or lemma == "zzz":
                        lemma = token
                    if not pos.strip() or not lemma.strip() or not token.strip():
                        print line
                    F.write(token+"\t"+pos+"\t"+lemma+"\n")
                    fullF.write(token+"\t"+pos+"\t"+lemma+"\n")
                except IndexError:
                    print line
                    continue
    trainF.close()
    devF.close()
    testF.close()
    return

def crm():
    # first rank the charters chronologically:
    SPLITTER = re.compile(r'p|r|a|c')
    meta = []
    with open("../data/original/annotated/crm-adelheid.txt", 'r') as oldF:
        for line in [line.strip() for line in oldF.readlines() if line.strip()]:
            if line.startswith("@"):
                try:
                    name = line.split()[3]
                    name = line.split(".")[0]
                    year = int("1"+SPLITTER.split(name)[1][:3])
                    meta.append([name,year])
                except:
                    pass
    meta.sort(key=itemgetter(1), reverse=False)
    lookup = {}
    for i in range(len(meta)):
        if str(i)[-1] == "9":
            lookup[meta[i][0]] = "DEV"
        elif str(i)[-1] == "0":
            lookup[meta[i][0]] = "TEST"
        else:
            lookup[meta[i][0]] = "TRAIN"
    with open("../data/uniform/annotated/crm-adelheid/full/crm-adelheid.3col", "w+") as fullF:
        fullF.write("\n")
    fullF = open("../data/uniform/annotated/crm-adelheid/full/crm-adelheid.3col", "a+")
    with open("../data/uniform/annotated/crm-adelheid/train/crm-adelheid_train.3col", "w+") as trainF:
        trainF.write("\n")
    trainF = open("../data/uniform/annotated/crm-adelheid/train/crm-adelheid_train.3col", "a+")
    with open("../data/uniform/annotated/crm-adelheid/dev/crm-adelheid_dev.3col", "w+") as devF:
        devF.write("\n")
    devF = open("../data/uniform/annotated/crm-adelheid/dev/crm-adelheid_dev.3col", "a+")
    with open("../data/uniform/annotated/crm-adelheid/test/crm-adelheid_test.3col", "w+") as testF:
        testF.write("\n")
    testF = open("../data/uniform/annotated/crm-adelheid/test/crm-adelheid_test.3col", "a+")
    F = None
    with open("../data/original/annotated/crm-adelheid.txt", 'r') as oldF:
        for line in [line.strip() for line in oldF.readlines() if line.strip()]:
            if line.startswith("@"):
                # determine file to write to
                name = line.split()[3]
                name = line.split(".")[0]
                try:
                    if lookup[name] == "TRAIN":
                        F = trainF
                    elif lookup[name] == "DEV":
                        F = devF
                    elif lookup[name] == "TEST":
                        F = testF
                    else:
                        print "Ooooops..."
                    F.write("@ begin_of_"+name+"\n")
                    fullF.write("@ begin_of_"+name+"\n")
                except KeyError:
                    continue
            else:
                try:
                    comps = WHITESPACE.split(line)
                    token, lemma, pos = comps[2], comps[3], comps[4]
                    token = clean_token(token)
                    token = token.replace("+", "~")
                    if len(token)<1 or len(lemma)<1 or len(pos)<3 or "..." in token:
                        continue
                    pos = pos.replace("#", "").replace("*", "") # remove indications we don't use...
                    lemma = lemma.replace("-", "").lower()
                    if lemma == "frans" and pos == "999":
                        lemma = token
                    if "+" in pos:
                        pos_comps = pos.split("+")
                        for i in range(len(pos_comps)):
                            pos_comps[i] = tag_converter.removeAdEnding(pos_comps[i])
                        pos = "+".join(pos_comps)
                    else:
                        pos = tag_converter.removeAdEnding(pos)
                    if lemma == "latijn" or lemma == "zzz":
                        lemma = token
                    if not pos.strip() or not lemma.strip() or not token.strip():
                        print line
                    F.write(token+"\t"+pos+"\t"+lemma+"\n")
                    fullF.write(token+"\t"+pos+"\t"+lemma+"\n")
                except IndexError:
                    print line
                    continue
    trainF.close()
    devF.close()
    testF.close()
    return


if __name__ == "__main__":
    tag_converter = TagConverter("../data/etc/TagConversionLookup.txt")
    cg_lit()
    relig()
    cgadmin()
    crm()
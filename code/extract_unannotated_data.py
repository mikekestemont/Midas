#!usr/bin/env python
# -*- coding: utf-8 -*-

import re, codecs, os

from TagConverter import *

from bs4 import BeautifulSoup as Soup
import bs4
from difflib import SequenceMatcher

text_tag = re.compile(r"<text>")
whitespace = re.compile(r"\s+")
brackets = re.compile(r"/\[.*?\]/")
numbers = re.compile(r"[0-9]+")
paragraph = re.compile(r'\¶')
separator = re.compile("(<C )|(<q>)")
underscore = re.compile("\_")
right = re.compile(r"\>\s+")
ABBR_TAGS = re.compile(r"(<A >)|(</A>)")
VN_TYPE = re.compile(r"<VN type=[0-9]+>")


def clean_token(dirtyT):
    punctuation = re.compile(r"\.|\(|\)|\"|\=|\)|\[|\]|\<|\>|\'|\{|\}|\\|\_|\@")
    cleanT = dirtyT.replace("&duitsekomma;", "/")
    cleanT = cleanT.replace("&period;", ".")
    cleanT = cleanT.replace("&comma;", ",")
    cleanT = cleanT.replace("&tilde;", "|")
    cleanT = cleanT.replace("&hyph;", "-")
    cleanT = cleanT.replace("&semi;", ";")
    cleanT = cleanT.replace("&colon;", ":")
    cleanT = cleanT.replace("&unreadable;", "?")
    cleanT = brackets.sub("", cleanT)
    cleanT = numbers.sub("", cleanT)
    cleanT = punctuation.sub("", cleanT)
    cleanT = paragraph.sub("", cleanT)
    cleanT = cleanT.encode("ascii" , "ignore")
    cleanT = whitespace.sub("", cleanT)
    return cleanT.lower().strip()

def clean_text(dirtyT):
    punctuation = re.compile(r"\?|\,|\.|\=|\!|\(|\)|\"|\&|\=|\*|\)|\[|\]|\<|\>|\'|\{|\}|\:|\;|\\|\/|\||\+]\-")
    cleanT = brackets.sub(" ", dirtyT)
    cleanT = numbers.sub(" ", cleanT)
    cleanT = punctuation.sub(" ", cleanT)
    cleanT = paragraph.sub(" ", cleanT)
    cleanT = cleanT.encode("ascii" , "ignore")
    return cleanT.lower().strip()

def cdrom():
    # now parse the cd-rom raw texts as we got in corrupt TEI-XML from the INL
    for textN in os.listdir("../data/original/unannotated/cdrom_xml/"):
        if not textN.endswith(".xml"):
            continue
        print textN
        with open("../data/original/unannotated/cdrom_xml/"+textN) as oldF:
            try:
                text = text_tag.split(oldF.read(), maxsplit=1)[1]
                soup = Soup(text)
                text = soup.get_text()
                text = clean_text(text)
                if not text.startswith("voor de tekst zie"):
                    with codecs.open("../data/uniform/unannotated/cdrom/"+str(textN)+".txt", "w+", "utf-8") as newF:
                        newF.write(text)
            except:
                pass
    # now parse the cd-rom raw texts from Brill (which we didn't get via the INL)
    # in the format as Lisanne downloaded them from the Cd-rom
    for textN in os.listdir("../data/original/unannotated/cdrom_txt/"):
        if not textN.endswith(".txt"):
            continue
        print textN
        with codecs.open("../data/original/unannotated/cdrom_txt/"+textN, "r+", "utf-8-sig") as oldF:
            words = [clean_token(w) for w in oldF.read().strip().split()]
            with codecs.open("../data/uniform/unannotated/cdrom/"+textN, "w+", "utf-8-sig") as newF:
                newF.write(" ".join(words))
    return

def lisanne():
    # parse the text's from Lisanne's corpus
    semicol = re.compile(r'[^"]{1}\;')
    orig_dir, dest_dir = "../data/original/raw/lisanne_own/", "../data/uniform/raw/lisanne_own/"
    for textN in sorted(os.listdir(orig_dir)):
        if not textN.endswith(".txt"):
            continue
        print textN
        with codecs.open(orig_dir+textN, "r", "utf-8-sig") as oldF:
            words = [clean_token(w) for w in oldF.read().strip().split()]
            with codecs.open(dest_dir+textN, "w+", "utf-8-sig") as newF:
                newF.write(" ".join(words))
    return

def relig(singleOutputFile):
    if singleOutputFile:
        orig_dir, destF = "../data/original/rich/relig/", "../data/uniform/rich/relig.txt"
        with open(destF, 'w+') as destF:
            for f in sorted(os.listdir(orig_dir)):
                print f
                if not f.endswith(".txt"):
                    continue
                tab = re.compile(r'\t')
                with open(orig_dir+f, 'r') as origF:
                    for line in [line.strip() for line in origF.readlines()]:
                        print line
                        print f
                        token, pos, lemma = tab.split(line)
                        if "+" in pos:
                            new_pos = []
                            for pos in pos.split("+"):
                                new_pos.append(tag_converter.removeAdEnding(pos))
                            pos = "+".join(new_pos)
                        else:
                            pos = tag_converter.removeAdEnding(pos)
                        destF.write("\t".join([token, pos, lemma])+"\n")
    else:
        orig_dir, dest_dir = "../data/original/rich/relig/", "../data/uniform/rich/relig/"
        for f in sorted(os.listdir(orig_dir)):
            if not f.endswith(".txt"):
                continue
            tab = re.compile(r'\t')
            with open(orig_dir+f, 'r') as origF:
                with open(dest_dir+f, 'w+') as newF:
                    for line in [line.strip() for line in origF.readlines()]:
                        token, lemma, pos = tab.split(line)
                        pos = tag_converter.removeAdEnding(pos)
                        newF.write("\t".join([token, pos, lemma])+"\n")
    return

def main():
    # create global converter: 
    tag_converter = TagConverter("../data/etc/TagConversionLookup.txt")
    cdrom()

if __name__ == "__main__":
    main()

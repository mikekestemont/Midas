#!usr/bin/env python
# -*- coding: utf-8 -*-

import re, codecs, os, itertools, subprocess, sys, shutil

from TagConverter import *

if not os.path.isdir("../workspace"):
	os.mkdir("../workspace")
if os.path.isdir("../workspace/rnnlm"):
	shutil.rmtree("../workspace/rnnlm")
os.mkdir("../workspace/rnnlm")

tab = re.compile(r"\s*\t+\s*")
whitespace = re.compile(r"\s+")
brackets = re.compile(r"/\[.*?\]/")
numbers = re.compile(r"[0-9]+")
paragraph = re.compile(r'\¶')

def clean_token(dirtyT):
	punctuation = re.compile(r"\?|\,|\.|\=|\!|\(|\)|\"|\&|\=|\*|\)|\[|\]|\<|\>|'|\{|\}|\:|\;|\\|\/|\||\-|\_|\@")
	cleanT = brackets.sub("", dirtyT)
	cleanT = numbers.sub("", cleanT)
	cleanT = punctuation.sub("", cleanT)
	cleanT = paragraph.sub("", cleanT)
	cleanT = cleanT.encode("ascii" , "ignore")
	cleanT = whitespace.sub("", cleanT)
	return cleanT.lower().strip()

LEFT_CONTEXT = 2
RIGHT_CONTEXT = 2

# create global converter: 
tag_converter = TagConverter("../data/etc/TagConversionLookup.txt")

focus_tokens = set() # the composite tokens we have seen verbatimly in the training materials
trainPOStags = set() # all the (splitted) pos-tags we have seen verbatimly in the training materials

trainPOS_lmF = open("../workspace/rnnlm/correctPOS.txt", 'w+')
nbestPOS_lmF = open("../workspace/rnnlm/nbestPOS.txt", 'w+') # nbest file on which we will actually run the rnnlm
nbestLEM_lmF = open("../workspace/rnnlm/nbestLEM.txt", 'w+') # nbest file which also contains the lemmas

def extract_training_file(path=""):
	with open(path, 'r') as F:
		print "Loading from "+path
		for i, line in enumerate(F):
			line = line.strip()
			try:
				token, pos, lemma = tab.split(line)
				token, lemma = clean_token(token), clean_token(lemma)
				if len(token) < 1 or len(lemma) < 1 or len(pos) < 1:
					continue
				pos_tags = []
				for p in [p for p in pos.split("+")]:
					trainPOStags.add(p)
					if p.startswith("Punc"):
						pos_tags.append("<utt>") # punctuation is the closest we are gonna get to verse endings in CG-lit...
					else:
						pos_tags.append(p)
				trainPOS_lmF.write(" <clitic> ".join(pos_tags)+" ") # indicate where 2 pos-tags were joined
			except IndexError:
				continue
	return

extract_training_file("../data/uniform/rich/admin/cg-admin.txt")
extract_training_file("../data/uniform/rich/admin/adelheid.txt")

trainPOS_lmF.close()

# split available data in train+valid for training the RNNLM
trainF, validF = open("../workspace/rnnlm/trainRNNLM.txt",'w+'), open("../workspace/rnnlm/validRNNLM.txt", 'w+')
with open("../workspace/rnnlm/correctPOS.txt", "r") as allF:
	words = whitespace.split(allF.read())
	print "Using "+str(len(words))+" words in total for the RNNLM:"
	trainW = words[:int((len(words)/100)*90)] 
	print "\t> Using "+str(len(trainW))+" words for training the RNNLM"
	validW = words[-int((len(words)/100)*10):] 
	print "\t> Using "+str(len(validW))+" words for validating the RNNLM"
	trainF.write(" ".join(trainW))
	validF.write(" ".join(validW))
trainF.close()
validF.close()

NR = 0
nr_nbest = 0

cglit_dir = "../data/uniform/rich/cg-lit/"
cglit_intermed_dir = "../data/uniform/rich/cg-lit-intermediate/"
cglit_reorder_dir = "../data/uniform/rich/cg-lit-reordered/"

for textN in sorted(os.listdir(cglit_dir)):
	if not textN.endswith(".fromdb.txt"):
		continue
	print textN
	verse_line = []
	with open(cglit_dir+textN, "r") as origF:
		oldLines = [line.strip() for line in origF.readlines()]#[:2000]
		lines = []
		for l in oldLines:
			if l.strip() == "<utt>":
				lines.append("<utt>")
			else:
				try:
					token, pos, lemma = tab.split(l)
					token, lemma = clean_token(token), clean_token(lemma)
					pos = whitespace.sub("",pos)
					if len(token) < 1 or len(lemma) < 1 or len(pos) < 1:
						continue
					if not "+" in pos:
						lines.append([token, [pos+"@"+lemma]])
					else:
						pos_comps, lem_comps = pos.split("+"), lemma.split("+")
						if len(pos_comps) != len(lem_comps):
							continue
						tags = ["@".join(combo) for combo in zip(pos_comps, lem_comps)]
						lines.append([token, tags])
				except ValueError:
					pass
	with open(cglit_intermed_dir+textN, "w+") as intermF:
		for i, line in enumerate(lines):
			if line == "<utt>":
				intermF.write("<utt>\n")
			else:
				if len(line[1]) == 1: # no comps, atomic token
					token = line[0]
					pos, lemma = line[1][0].split("@")
					intermF.write(("\t".join([token, pos, lemma]))+"\n")
				else:
					NR+=1
					nr_nbest+=1
					token = line[0]
					intermF.write(token+"\t"+str(nr_nbest)+"\n")
					skeleton = []
					for c in range(-(LEFT_CONTEXT), (RIGHT_CONTEXT)+1): # specify the context window
						if (i+c >= 0) and (i+c < len(lines)): # check whether there are context tokens
							l = lines[i+c]
							if l == "<utt>":
								skeleton.append([["<utt>"]])
							else:
								tags = []
								for p in l[1]:
									tags.append(p)
								skeleton.append(list(itertools.permutations(tags)))
						else:
							skeleton.append([["<unk>"]])
					for prod in itertools.product(*skeleton):
						all_s = ""
						for position in prod:
							if len(position) == 1:
								all_s+=str(position[0])+" "
							elif len(position) > 1:
								all_s+= (" <clitic> ".join(position))+" "
						nbestLEM_lmF.write(str(nr_nbest)+" "+all_s+"\n")
						comps = []
						for c in all_s.split(" "):
							comps.append(c.split("@")[0])
						nbestPOS_lmF.write(str(nr_nbest)+" "+(" ".join(comps))+"\n")
nbestPOS_lmF.close()
nbestLEM_lmF.close()
#print(nr_nbest)

WD = "/home/mike/GitRepos/Midas/workspace/rnnlm/"

subprocess.call("~/rnnlm/rnnlm rnnlm -train "+WD+"trainRNNLM.txt -rnnlm "+WD+"modelRNNLM.txt -valid "+WD+"validRNNLM.txt -hidden 50 -bptt 4", shell=True)
subprocess.call("~/rnnlm/rnnlm rnnlm -test "+WD+"nbestPOS.txt -rnnlm "+WD+"modelRNNLM.txt -nbest > "+WD+"rnn_scores.txt", shell=True)

scores = []
with open(WD+"rnn_scores.txt", 'r') as F:
	for line in F:
		nr = line.strip()
		if len(nr) == 0: continue
		if nr[0].isalpha(): continue
		scores.append(abs(float(nr)))

best_orders = []
c = 0
id = "1"
lowest_score = sys.maxint
current_best = None
with open(WD+"nbestLEM.txt", 'r') as F:
	candidates = []
	for line in F:
		line = line.strip()
		if len(line) == 0:
			continue
		items = line.split(" ")
		current_id, ordering, score = items[0], items[1:], scores[c]
		if current_id == id:
			if score < lowest_score:
				current_best = ordering
				lowest_score = score
		else:
			best_orders.append(current_best)
			id = current_id
			lowest_score = score
			current_best = ordering
		c+=1
	if score < lowest_score:
		current_best = ordering
		lowest_score = score
	best_orders.append(current_best)
#print len(best)

for textN in sorted(os.listdir(cglit_intermed_dir)):
	if not textN.endswith(".fromdb.txt"):
		continue
	with open(cglit_intermed_dir+"/"+textN, 'r') as intermF:
		with open(cglit_reorder_dir+"/"+textN, "w+") as finalF:
			for line in intermF:
				line = line.strip()
				if line == "":
					continue
				if line == "<utt>":
					finalF.write(line+"\n")
				elif len(line.split()) == 3:
					finalF.write(line+"\n")
				elif len(line.split()) == 2: # one of the items we have to reorder
					print line.split()
					token, index = line.split()
					best_order = best_orders[int(index)-1]
					print best_order
					# extract focus items:
					# first remove left context:
					left = LEFT_CONTEXT
					while left > 0:
						if best_order[0].strip() == "<clitic>":
							del best_order[0]
						else:
							if best_order[1] != "<clitic>":
								left-=1
							del best_order[0]
						print best_order
						print left
					posses, lemmas = [], []
					print best_order
					for index, item in enumerate(best_order):
						if item.strip() == "<clitic>":
							continue
						print item
						pos, lem = item.split("@")
						posses.append(pos)
						lemmas.append(lem)
						if best_order[index+1] != "<clitic>":
							break
					pos = "+".join(posses)
					lem = "+".join(lemmas)
					print "\t".join([token, pos, lem])
					finalF.write("\t".join([token, pos, lem])+"\n")
					print "=========================="

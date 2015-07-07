[*very much* UNDER DEVELOPMENT]

Midas
=====

### Introduction
This repository holds the code for "Midas" (Middle Dutch Annotation System), a Tagger-Lemmatiser for Middle Dutch.  While Midas has been originally developed to deal with medieval Dutch, it is largely language-independent and can be applied to other (historic) languages such as medieval Latin or Old French. Midas provides functionality for tokenization, part-of-speech tagging and lemmatization, with a heavy bias towards language which show a considerable amount of orhtographic variation in spelling and spacing. Midas is written in pure Python (>= py2.7 or py3) and has been tested on UNIX-like systems. Via keras and theano, Midas makes heavy use of neural networks for its language modeling: luckily, training the tagger-lemmatizer can accelerated by running Midas on the GPU instead of the CPU.

### Data format
All input files should be encoded in UTF-8. Midas expects annotated training data to have the following, three-column format:

```
@ begin_of_text.txt
ambrosius	N(prop)	ambrosius
ende	Conj(coord)	en
iacob	N(prop)	jacob
van	Adp()	van
uitri	N(prop)	vitry

ende	Conj(coord)	en
isidorus	N(prop)	isidorus
dar~bi	PronAdv(dem)	daarbij

nomic	V(fin,pres,lex)+Pron(pers,1,sing)	noemen+ik
iv	Pron(pers,2,plu)	gij
dese	Pron(dem)	deze
bi	Adp()	bij
namen	N(sing)	naam
```

A normal line should contain the original token, the part-of-speech tag and the lemma, separated by tabs. The beginning of a new document can be encoded as "@ begin_of_text.txt". Empty newlines (`\n\n`) can be used to indicate utterance boundaries, e.g. to mark verse endings in medieval poetry. If consecutive tokens in the original input, had to be concatenated to assign a lemma to them (e.g. `dar~bi` in the example above), the concatenation can be marked using a tilde and, if needed, a tokenizer can be trained to learn and reproduce this behaviour. Due to cliticization phenomena, sometimes composite tags are assigned to words (e.g. noemen+ik); Midas considers as these atomic tags. Midas is agnostic with respect to the specific tag or lemma set used: any system can be used as long as it is consistent.

With respect to unannotated data (used for pretraining), Midas simply expects utf8-encoded files, respecting the original spacing between tokens and using empty lines to mark boundaries between utterances:

```
ambrosius ende iacob van uitri

ende isidorus dar bi

nomic iv dese bi namen
```

### Running midas
Midas can be used in the following modes: "tag", "test" and "train". Its configuration and hyperparameters can be set using a standard config file. Previously trained models can be saved via pickling and reused for tagging or testing. Run midas from the command line:

```
>>> python midas.py train config.txt my_model
>>> python midas.py tag config.txt my_model
>>> python midas.py test config.txt my_model
```

To enable GPU acceleration, add something like:
```
>>> THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python midas.py train config.txt my_model
```


### Dependencies
Midas mainly depends on scikit-learn, keras (and thus theano). If you want to use theano's support GPU-acceleration (which comes highly recommended for larger data sets), you will have to properly install Nvidia’s CUDA.





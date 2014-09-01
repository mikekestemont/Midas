#!/usr/bin/env python

import re

class TagConverter(object):
	def __init__(self, lookupfile):
		self.lookupfile = lookupfile
		self.init_lookupdicts(lookupfile)
		return

	def init_lookupdicts(self, lookupfile):
		self.Ad2NumDict = {} # lookup dict to convert Ad(elheid) tags to Num(eric) tags
		self.Num2AdDict = {} # lookup dict to convert Num(eric) tags Ad(elheid) tags
		with open(lookupfile, 'r') as f:
			for line in f.readlines():
				line = line.strip()
				if len(line) < 1 or line.startswith("#"):
					continue
				Ad, Num = line.split("/")[:2]
				if Num.strip() == "" and Ad.strip() == "":
					continue
				if Num.strip() == "":
					Num = "NA"
				if Ad.strip() == "":
					Ad = "NA"
				self.Ad2NumDict[Ad] = Num
				self.Num2AdDict[Num] = Ad
		return

	def Ad2Num(self, Ad):
		try:
			return self.Ad2NumDict[Ad]
		except KeyError:
			for k in self.Ad2NumDict:
				if k.startswith(Ad[:2]):
					return self.Ad2NumDict[k]
		return "???"


	def Num2Ad(self, Num):
		if Num in self.Num2AdDict:
				return self.Num2AdDict[Num]		
		else:
			for k in self.Num2AdDict:
				if k.startswith(Num[:2]):
					return self.Num2AdDict[k]
		print Num
		return "???"

	def removeAdEnding(self, Ad):
#		if Ad not in self.Ad2NumDict.keys():
#			return "NA"
		try:
			main_tag, rest = Ad.split("(")
			subtags = rest.split(',')
			subtags = ",".join([tag for tag in subtags if not tag.startswith(("form", "unclear"))])
			final_tag = main_tag+"("+subtags
			if not final_tag.endswith(")"):
				final_tag+=")"
			return final_tag
		except:
			print Ad
			return Ad
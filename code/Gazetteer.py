#!usr/bin/env python

class Gazetteer(object):
    def __init__(self, path2gazetteers=""):
        """
        >>> G = Gazetteer("../data/etc/gazetteers.txt")
        >>> print G.lookup_token("kaerl")
        >>> print G.lookup_token("mike")
        """
        self.lookupD = {}
        for line in [line.strip() for line in open(path2gazetteers, 'r')]:
            if line:
                key, val = line.split(">")
                self.lookupD[key.strip().lower()] = val.strip().lower()
        return

    def lookup_token(self, token=""):
        try:
            return self.lookupD[token]
        except KeyError:
            return "no_name"

    def __str__(self):
        info = ""
        for k in self.lookupD:
            info+= (k + " : " + self.lookupD[k] + "\n")
        return info

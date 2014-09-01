#!usr/bin/env python

from difflib import SequenceMatcher

class Transliterator():
    """
    Class which relies on difflib's SequenceMatcher to calculate the optimal alignment of two strings.
    """
    def __init__(self, context=3):
        """
        Constructor
        """
        self.context = context
        return

    def transliterate(self, t1="token_a", t2="token_b"):
        """
        Create an alignment between two strings. Returns a dict that represent each edit operation in context.
        E.g.
        >>> T = Transliterator(4)
        >>> T.transliterate(t1='schepenen', t2="scepen")
        """
        feature_dicts = []
        labels = []
        s = SequenceMatcher(None, t1, t2)
        c = 0
        leftover = ""
        # determine empty slots in t1:
        t1New = ""
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag == "insert":
                t1New+="X"
            else:
                t1New+=t1[i1:i2]
        for tag, i1, i2, j1, j2 in s.get_opcodes():
#           print("%7s a[%d:%d] (%s) b[%d:%d] (%s)" % (tag, i1, i2, t1[i1:i2], j1, j2, t2[j1:j2]))
            if tag == "equal":
                for source_char, target_char in zip(t1[i1:i2], t2[j1:j2]):
                    labels.append(leftover+target_char)
            elif tag == "replace":
                appended = False
                for source_char in t1[i1:i2]:
                    if not appended:
                        labels.append(t2[j1:j2])
                        appended = True
                    else:
                        labels.append("X")
            elif tag == "delete":
                for source_char in t1[i1:i2]:
                    labels.append("X")
            elif tag == "insert":
                labels.append(t2[j1:j2])
        t1 = t1New
#       print(list(t1New))
#       print(str(labels))
        for index, focus_char in enumerate(t1):
            D = {}
            # right context:
            right = []
            c = 1
            while (index+c) < len(t1):
                if len(right) >= self.context:
                    break
                D["r+"+str(c)] = t1[index+c]
                right.append(t1[index+c])
                c+=1
            while len(right) < self.context:
                D["r+"+str(c)] = "="
                right.append("=")
                c+=1
            # left context:
            left = []
            c = 1
            while (index-c) >= 0:
                if len(left) >= self.context:
                    break
                left.append(t1[index-c])
                D["l-"+str(c)] = t1[index-c]
                c+=1
            left.reverse()
            while len(left) < self.context:
                D["l-"+str(c)] = "="
                left.insert(0, "=")
                c+=1
            D["focus"] = focus_char
            #print(D)
            #print(left),
            #print(" | "+focus_char+" | "),
            #print(right),
            #print(" : "+str([labels[index]]))
            #print "*********"
            feature_dicts.append(D)
        return (feature_dicts, labels)

#T = Transliterator(4)
#T.transliterate(t1='schepenen', t2="scepen")

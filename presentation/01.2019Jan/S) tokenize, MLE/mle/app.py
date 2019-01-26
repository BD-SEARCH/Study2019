# -*- coding: utf-8 -*-
import nltk
from nltk.util import ngrams
from nltk.corpus import alpino

# print(alpino.words())

# unigrams = ngrams(alpino.words(),1)
# quadgrams = ngrams(alpino.words(),4)
# for i in unigrams:
#     print(i)

# for i in quadgrams:
#     print(i)

from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords, webtext
from nltk.metrics import BigramAssocMeasures

# tokens = [t.lower() for t in webtext.words('grail.txt')]
# words = BigramCollocationFinder.from_words(tokens)
# print(words.nbest(BigramAssocMeasures.likelihood_ratio, 15))

set = set(stopwords.words('english'))
stops_filter = lambda w:len(w)<3 or w in set

tokens = [t.lower() for t in webtext.words('grail.txt')]

words = BigramCollocationFinder.from_words(tokens)
words.apply_word_filter(stops_filter)
print(words.nbest(BigramAssocMeasures.likelihood_ratio, 10))
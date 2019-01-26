# -*- coding: utf-8 -*-
# replacement
import re
import nltk
from nltk.corpus import wordnet

replacement_patterns = [
    ("i\'m", "i am"),
    ("(\w+)\'ll", "\g<1> will"),
    ("(\w+)\'r", "\g<1> not"),
    ("(\w+)\'ve", "\g<1> have"),
    ("(\w+)\'s", "\g<1> is"),
    ("(\w+)\'re", "\g<1> are"),
    ("(\w+)\'d", "\g<1> would"),
    ("(\w+)n\'t", "\g<1> not")
]

# 단어 풀어쓰기
class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s

# 중복되는 단어 수정 ex) lottttt -> lot
class RepeatReplacer(object):
    def __init__(self):
        self.repeat_regexp = re.compile(r"(\w*)(\w)\2(\w*)")
        self.repl = r"\1\2\3"

    def replace(self, word):
        if wordnet.synsets(word): return word
        repl_word = self.repeat_regexp.sub(self.repl, word)
        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word

# 대체 가능한 단어 대체
class WordReplacer(object):
    def __init__(self, word_map):
        self.word_map = word_map
    def replace(self, word):
        return self.word_map.get(word, word)
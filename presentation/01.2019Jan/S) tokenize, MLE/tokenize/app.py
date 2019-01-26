# -*- coding: utf-8 -*-
import nltk
from replacePatterns import RegexpReplacer, RepeatReplacer, WordReplacer
from nltk.tokenize import word_tokenize

# replacer = RegexpReplacer()
# replacer = RepeatReplacer()
replacer = WordReplacer({"congrats":"congratulations"})

text = "Don't hesitate to ask questions"

# case1. 단어 풀기
# print(word_tokenize(text))
# print(word_tokenize(replacer.replace(text)))

# case2. 중복단어 수정
# print(replacer.replace("happppppy"))

# case3. 동음이의어 대체
print(replacer.replace("congrats"))
print(replacer.replace("maths"))
from __future__ import print_function
# -*- coding: utf-8 -*-
import nltk
from nltk.metrics import *

def _edit_dist_init(len1, len2):
    A = []
    for i in range(len1):
        A.append([0] * len2)

    # (i,0), (0,j) 채우기
    for i in range(len1):
        A[i][0] = i
    for j in range(len2):
        A[0][j] = j

    return A

def _edit_dist_step(A, i, j, s1, s2, transpositions=False):
    c1 = s1[i-1]
    c2 = s2[j-1]

    a = A[i-1][j] + 1 # s1에서 skip
    b = A[i][j-1] + 1 # s2에서 skip
    c = A[i-1][j-1] + (c1!=c2) # 대체
    d = c+1 # X select

    if transpositions and i>1 and j>1:
        if s1[i-2] == c2 and s2[j-2] == c1:
            d = A[j-2][j-2] + 1

        A[i][j] = min(a,b,c,d)

def edit_distance(s1, s2, transpositions=False):
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)

    for i in range(len1):
        for j in range(len2):
            _edit_dist_step(lev, i+1, j+1, s1, s2, transpositions=transpositions)
    return lev[len1][len2]

def jacc_sim(query, document):
    first = set(query).intersection(set(document))
    second = set(query).union(set(document))
    return len(first)/len(second)


# print(edit_distance("suggestion", "calculation"))
x = set([10,20,30,40])
y = set([20,30,60])
print(jacc_sim(x,y))
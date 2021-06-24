'''
=======================================================================
This file uses the Mandarin word2vec and returns a similarity analysis for
various words.

This Python script reads one file, the es.bin file for a pre-trained
word2vec. .bin files for other languages can be downloaded here:
https://github.com/Kyubyong/wordvectors

Visualization code from here + CJK code from Prof. Coto-Solano:
https://stackoverflow.com/questions/43776572/visualise-word2vec-generated-from-gensim
=======================================================================
'''

# Import libraries to run word2vec and make charts
import pickle
from gensim.models import Word2Vec
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def print_ls(ls):
  print("-----------------------------")
  print("{:<13}{:<17}".format("Word","Similarity"))
  print("-----------------------------")
  for val in ls:
    print(val)
  print("-----------------------------\n")

model = Word2Vec.load('zh/zh.bin') # for some reason genisim < 4.0.0 works
print('女人:')
print_ls(model.most_similar(positive=['女人'], topn=25)) # most similar to woman
print('男人:')
print_ls(model.most_similar(positive=['男人'], topn=25)) # most similar to man
print('皇后-男人+女人:')
print_ls(model.most_similar(positive=['皇后', '女人'], negative=['男人'], topn=25)) # king-man+woman
print('男人+家:')
print_ls(model.most_similar(positive=['男人', '家'], topn=25)) # man+home
print('女人+家:')
print_ls(model.most_similar(positive=['女人', '家'], topn=25)) # woman+home

# t-SNE chart for [ 'man', 'woman', 'king', 'queen', 'child', 'boy', 'girl' ]
words = ['男人', '女人', '皇帝', '皇后', '小孩', '男孩', '女孩']
eng = [ 'man', 'woman', 'king', 'queen', 'child', 'boy', 'girl' ]

arr = np.empty((0,300), dtype='f')
for word in words:
    vec = model.wv[word]
    arr = np.append(arr, np.array([vec]), axis=0)

tsne = TSNE(n_components=2) # instantiate t-SNE chart
np.set_printoptions(suppress=True)

Y = tsne.fit_transform(arr)
x_coords = Y[:, 0]
y_coords = Y[:, 1]

plt.scatter(x_coords, y_coords) # create chart
for label, x, y in zip(eng, x_coords, y_coords):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.show() # display plot

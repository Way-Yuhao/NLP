'''
Created on Mar 27, 2019

@author: LiuYuhao
'''
from numpy import sort

''' ROADMAP
Input all files and store in separate classes
Preprocessing
Tokenization
Build bag-of-words representation of each documents using collection.counter
tf-idf
'''
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import norm
from collections import Counter

docs = []
templ = []


def main():
    #method variables
    global docs
    global templ
    global occurrance
    global wordType
    
    docs = []
    templ = []
    stopwords = ["a", "the", "of", "with"]
    os.chdir('news')
    index = 1;
    
    while index <= 511:
        # open file
        file = open("{:03d}.txt".format(index))
        lines = file.read() # convert all to lowercase; type = str
        # tokenize
        tokens = lines.split()
        # pre-processing
        ct = Counter()
        for token in tokens:
            ct[token]+= 1
        docs.append(ct)
        file.close()
        index+=1
    print("----Finished building all docs----") 
       
    # combine all docs to obtain dictionary
    corpus = Counter()
    for doc in docs:
        corpus += doc
    occurrance = sum(corpus.values())
    wordType =  len(corpus)
    print("Total occurrence = {}".format(occurrance))
    print("Word types = {}".format(wordType))
    
    # building corpus template
    templ = []
    for x, y in list(corpus.most_common()):
        templ.append(x)
    
    print('----Ranking Top 20 word types----')
    rankedCorpus = corpus.most_common(20)
    index = 1;
    for word in rankedCorpus:
        print("{}) {}".format(index, word))
        index+= 1;
        
    # plotting graph
    print('---plotting graph----')
    x = range(1, wordType + 1)
    y = sorted(list(corpus.values()), reverse = True)
    logx = np.log10(x)
    logy = np.log10(y)
#     plt.scatter(x, y)
#     plt.show()
#     plt.scatter(logx, logy)
#     plt.show()
    
    # TF-IDF processing
    v_098= getBoWVectorFor(docs[97])
    v_297= getBoWVectorFor(docs[296])
    v_098_tf= getTfidfVectorFor(docs[97], 'contract')
    v_297_tf= getTfidfVectorFor(docs[296])
    
    print('----TOP 10 TF-IDF in 098.txt----')
    sortedTuple = sorted(list(zip(templ, v_098_tf)), key = lambda tup: tup[1], reverse = True)[0:10]
    index = 0
    for entry in sortedTuple:
        print('{}) {}'.format(index+ 1, entry))
        index += 1
        
    print('----Calculating Cosine Similarity----')
    print('Between file 098.txt and 297.txt:')
    print('Cosine Similarity using BoW: {}'.format(calCosSim(v_098, v_297)))
    print('Cosine Similarity using tf-idf: {}'.format(calCosSim(v_098_tf, v_297_tf)))
    print('------------------------------')
          

def calCosSim(v1, v2):
    cos_sim = dot(v1, v2)/(norm(v1)*norm(v2))
    return cos_sim

def getBoWVectorFor(doc):
    token_count = sum(doc.values())
    v = [0] * wordType
    for w in doc:
        c = doc.get(w)
        v[templ.index(w)] = c / token_count
    return v
        
def getTfidfVectorFor(doc, checkWord = None):
    max_c = doc.most_common(1)[0][1]
    # print(max_c)
    v = [0] * wordType
    for w in doc:
        c = doc.get(w)
        tf_w = c / max_c
        idf_w = math.log(511 / docsContainW(w), 10)
        v[templ.index(w)] = tf_w * idf_w
        if (checkWord != None and w == checkWord):
            print('tf = {}, idf = {}'.format(tf_w, idf_w))
            
    return v
        
def docsContainW(w):
    count = 0
    for doc in docs:
        if doc.__contains__(w):
            count += 1
    return count
        
if __name__ == '__main__':
    main()
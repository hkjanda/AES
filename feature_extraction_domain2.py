 # -*- coding: utf-8 -*-
from __future__ import division
from nltk.corpus import wordnet as wn
#a= time.time()
from itertools import izip_longest
# from pywsd import disambiguate
from semantic_nets_corpus_statistics import length_between_synsets, hierarchical_distance
from frequency_in_wn import return_max_frequency_synset
import numpy as np
import pandas as pd
import json
from pymagnitude import *
import requests
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import pprint
import autocorrect
import string
from nltk import FreqDist
from autocorrect import spell
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("english")
#from negatives import process_negation
from length_grammar import dependency_parser
from length_grammar import dependency_parser
from grammar import grammar_fn
import nltk
import re
from spellchecker import SpellChecker
import spacy
import numpy
import en_core_web_sm
spell = SpellChecker()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
import math
from array import*
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from itertools import combinations
from eigen import eigen1
import sqlite3
import csv
import enchant
chkr = enchant.Dict("en_US")
data = {}
similarity_values = {}
div_index = 0

nlp = en_core_web_sm.load()
nlp=spacy.load('en')

# con = sqlite3.connect(":memory:")
# con.text_factory = bytes
# Global variables

vectors = Magnitude("/home/compute/Downloads/wiki-news-300d-1M.magnitude")


def grouper(iterable, n,fillvalue=None):
    """
    grouper is used for reading n lines from a file altogether
    We use n=3 to read 3 lines from the file containing sentences examples and their previously established similarity value
    :param iterable: file with examples
    :param n: number of lines to be read simultaneously
    :param fillvalue: separator
    :return: n number of strings
    """
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)


# def tag_tokens(s):
#     """
#     Disambiguate the sentence and form a list of tagged words
#     :param s: sentence
#     :return:  list of words with appropriate synset and part of speech
#     """
#     list_of_tagged_tokens = disambiguate(s)
#     list_of_tagged_tokens = [i for i in list_of_tagged_tokens if i[1] is not None]
#     # print list_of_tagged_tokens
#     #for ele in list_of_tagged_tokens:
#         #print ele[1],":",ele[1].definition()
#     #print "\n\n"
#     #print type(list_of_tagged_tokens[0])
#     #print list_of_tagged_tokens
#     return list_of_tagged_tokens



def word2vec(w1,w2):
    
    res = vectors.similarity(w1,w2)
    result=float((res+1)/2)
    return result



def word_order_sim(vector_1, vector_2):
    l1 = len(vector_1)
    l2 = len(vector_2)
    l = max(l1, l2)
    if l2 > l1:
        order_vector_1 = np.arange(1, l)
        order_vector_2 = np.zeros(l)
    else:
        order_vector_2 = np.arange(1, l)
        order_vector_1 = np.zeros(l)




def grouper(iterable, n,fillvalue=None):
    """
    grouper is used for reading n lines from a file altogether
    We use n=3 to read 3 lines from the file containing sentences examples and their previously established similarity value
    :param iterable: file with examples
    :param n: number of lines to be read simultaneously
    :param fillvalue: separator
    :return: n number of strings
    """
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)


def tag_tokens(s):
    """
    Disambiguate the sentence and form a list of tagged words
    :param s: sentence
    :return:  list of words with appropriate synset and part of speech
    """
    list_of_tagged_tokens = disambiguate(s)
    list_of_tagged_tokens = [i for i in list_of_tagged_tokens if i[1] is not None]
    # print list_of_tagged_tokens
    #for ele in list_of_tagged_tokens:
        #print ele[1],":",ele[1].definition()
    #print "\n\n"
    #print type(list_of_tagged_tokens[0])
    #print list_of_tagged_tokens
    return list_of_tagged_tokens

def max_freq_info(list_of_tagged_tokens):
    """
    Get the synset having maximum frequency in a corpus, in this case corpus is wordnet
    :param list_of_tagged_tokens: tokens(tagged words) from tag_tokens
    :return: list of words and corresponding synset with maximum frequency sense from corpus
    """
    tagged_frequency_list = []
    for token in list_of_tagged_tokens:
        pos = str(token[1]).split('.')[1]
        # print token
        max_freq = return_max_frequency_synset(token[0], pos)
        tagged_frequency_list.append([token, max_freq])
    return tagged_frequency_list

def word_similarity(w1, w2):
    """
    Word similarity between words w1 and w2, returns a numeric value between 0 and 1
    :param w1: word 1
    :param w2: word 2
    :return: semantic similarity between word 1 and word 2
    """
    #print w1,w2,length_between_synsets(w1,w2)
    #print w1,w2,hierarchical_distance(w1,w2)
    return length_between_synsets(w1, w2) * hierarchical_distance(w1, w2)


def form_value_vector_tfidf(d1, d2):
    
    i=0
    #tokens_s1 = nltk.word_tokenize(d1)
    #tokens_s2 = nltk.word_tokenize(d2)

    
    tokens_s1 = [word.lower() for word in d1]
   
    tokens_s2 = [word.lower() for word in d2]
    

    stop_words = set(stopwords.words('english'))
    filtered_s1 = [w for w in tokens_s1 if not w in stop_words]
    #print filtered_s1
    filtered_s2 = [w for w in tokens_s2 if not w in stop_words]
    #print filtered_s2
    l1=len(filtered_s1)
    l2=len(filtered_s2)
    length = max(l1, l2)
    semantic_vector = np.zeros(length)
    semantic_vector.fill(0)
    for w1 in filtered_s1:
      s1 = 0
      #w1=spell(w1)
      temp_sim_list=list()
 
      for w2 in filtered_s2:
        
        #w2=spell(w2)
        s_sim= word2vec(w1,w2)
        #print w1
        #print w2
        #print s_sim
        temp_sim_list.append(s_sim)
        s1= max(temp_sim_list)
      semantic_vector[i] = s1
      i=i+1
    return semantic_vector

def form_value_vector(d1, d2):
    """
    form a semantic vector for sentences, d1 and d2 are the list of tagged words for s1 and s2 respectively
    :param d1:
    :param d2:
    :return:
    """
    #code to check spelling and rueturn number of wrong spells

    # print d1
    # print d2
    i=0
    #tokens_s1 = nltk.word_tokenize(d1)
    #tokens_s2 = nltk.word_tokenize(d2)

    tokens_s1=(tokenizer.tokenize(d1))
    tokens_s1 = [word.lower() for word in tokens_s1]
    tokens_s2=(tokenizer.tokenize(d2))
    tokens_s2 = [word.lower() for word in tokens_s2]
    

    stop_words = set(stopwords.words('english'))
    filtered_s1 = [w for w in tokens_s1 if not w in stop_words]
    #print filtered_s1
    filtered_s2 = [w for w in tokens_s2 if not w in stop_words]
    #print filtered_s2
    l1=len(filtered_s1)
    l2=len(filtered_s2)
    length = max(l1, l2)
    semantic_vector = np.zeros(length)
    semantic_vector.fill(0)
    for w1 in filtered_s1:
      s1 = 0
      #w1=spell(w1)
      temp_sim_list=list()
 
      for w2 in filtered_s2:
        
        #w2=spell(w2)
        s_sim= word2vec(w1,w2)
        #print w1
        #print w2
        #print s_sim
        temp_sim_list.append(s_sim)
        s1= max(temp_sim_list)
      semantic_vector[i] = s1
      i=i+1
    return semantic_vector

def word_order_sim(vector_1, vector_2):
    l1 = len(vector_1)
    l2 = len(vector_2)
    l = max(l1, l2)
    if l2 > l1:
        order_vector_1 = np.arange(1, l)
        order_vector_2 = np.zeros(l)
    else:
        order_vector_2 = np.arange(1, l)
        order_vector_1 = np.zeros(l)
#compute for imp words
def similarity_tfidf(vector_1, s1 ,s2):
    a = np.array(vector_1)
    mean_tfidf=np.mean(a)
    return mean_tfidf


def similarity(vector_1, vector_2, s1 ,s2):
    """
    Sentence similarity
    vector_1 and vector_2 are semantic vectors from form_value_vector
    :param vector_1: semantic vector for sentence 1
    :param vector_2: semantic vector for sentence 2
    """
    global similarity_values
    global div_index
    global data
    count = 0
    shift=0
    ###########################################################################################
    # print vector_1
    # print vector_2
    # print np.dot(vector_1,vector_2.T)
    # word_order_index=(1.0-DELTA)*word_order_similarity(s1,s2)
    # print "WOS:",word_order_index

    # print "np.dot(vector_1, vector_2)",np.innera(vector_1, vector_2)
    # print "np.linalg.norm(vector_1):",np.linalg.norm(vector_1)
    # print "np.linalg.norm(vector_2):",np.linalg.norm(vector_2)
    # print "(np.linalg.norm(vector_1) * np.linalg.norm(vector_2)):",(np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
    # print "DELTA*np.dot(vector_1, vector_2):",DELTA*np.dot(vector_1, vector_2)
    # print "ORIGINAL:",np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
    # print vector_1
    # print vector_2
    ############################################################################################
    sim = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
    #print sim
    for index in vector_1:
        if index > 0.80:
            count = count + 1
    for index in vector_2:
        if index > 0.80:
            count = count + 1

    # print "Divided by vec size:",sim/(vector_1.size)
    # print "Divided by vec/2:",sim/((vector_1.size)/2)
    if count > 2:
        sim = sim / (count / 1.8)
        if sim > 1.0:
            sim=1.0
        #print sim
        #negation = process_negation(s1,s2)
        #print "nenation",negation
        #if negation == 0.0:
            #sim=sim/1.5
            #pass
        #print sim

    else:
        try:
            sim = sim / (float(vector_1.size) / 2)
            #print ("inside try",sim)
        except:
            sim=sim

    if sim> 1.0:
            sim=1.0
        #negation = process_negation(s1, s2)
        #print "nenation" , negation
        #if negation == 0.0:
            #sim=sim/1.5
            #pass
        #print "after negation:",sim
        #print sim
        # similarity_values.dumps(data)
        #print similarity_values
        # return similarity_values
    #print sim
    length_s1=len(s1.split())
    length_s2=len(s2.split())
    length_difference=abs(length_s1-length_s2)
    if length_difference==0:
        pass
    else:
        try:
            shift=0.10*np.log(length_difference+1)
            #print "shift",shift
        except:
            shift=0.0

    dependency=dependency_parser(s1,s2)
    #print dependency
    if sim>0.85:
        if "no" in s1.split(" ") or "no" in s2.split(" ") or "nobody" in s1.lower().split(" ") or "nobody" in s2.lower().split(" "):
            #print("%.2f" % (sim-dependency-shift))
            sim=sim-dependency-shift
            if shift==0.0:
                sim=0.75

        else:
            #print("%.2f" % (sim-dependency))
            sim=sim-dependency-shift #chnged
            pass
    else:
        #print("%.2f" % (sim-shift))
        if sim>0.5:
            sim=sim-shift

    if length_s1==length_s2:
        missing1 = [x for x in s1.split(" ") if x not in s2.split(" ")]
        #print missing1
        missing2 = [x for x in s2.split(" ") if x not in s1.split(" ")]
        #print missing2
        #compare synsets of missing1 and missing2 and if equal then sim=1 else sim=0.7
        if len(missing1) == 1:
            syn1=wn.synsets(missing1[0])
            syn2=wn.synsets(missing2[0])
            for syn1_1 in syn1:
                for syn2_1 in syn2:
                    if word_similarity(syn1_1,syn2_1)>0.8:
                        sim=0.95
                        break
                if sim==0.95:
                    break
                sim=0.7
                if s1!=s2:
                    sim=0.5


    if "no" in s1.split(" ") or "no" in s2.split(" ") or "nobody" in s1.split(" ") or "nobody" in s2.split(" "):
        sim = 0.7-shift
    if s1==s2:
        sim=1.0
    #print (abs(sim))
    #data[''] = abs(sim)
    #similarity_values = json.dumps(data)
    #print similarity_values
    return abs(sim)

def syntax_results(file_name):

    # text1 = open(file_name,'r')#reading essay from input
    text_sents = sent_tokenize(file_name) #tokenize the sentences
    count =0
    #number of unique pos tags
    list= nltk.pos_tag(word_tokenize(file_name))
    pos_unique= len(set([x[1] for x in list]))
    #text_sents_clean=[remove_string_special_characters(s) for s in text_sents]
    
    
    
    
    slist = []
    lines=0
    words=0
    characters=0
    # with open(file_name) as fp:
        
            
    #split on fullstop
    sentences = file_name.split('.')
    for sentence in sentences:
        try:
            if sentence:
                slist.append(sentence)
        except BaseException:
            pass
    # print (slist)
    # with open(file_name) as fp: #count number of words and characters and average length of words
    #     for line in fp:
    wordslist=file_name.split()
    # lines=lines+1
    words_len=words+len(wordslist) 
    characters += sum(len(word) for word in wordslist)
         

    print("Number of words:")
    print(words)
    print("number of characters:")
    print(characters)

    count = 0

    for words in wordslist:
        if chkr.check(words) is False:
            count = count + 1

    print 'number of misspelled errors'
    print(count)

    #print ("average length of words")
    #print average
    
    # File = open(file_name) #open file
    # lines = File.read() #read all lines
    sentences = nltk.sent_tokenize(file_name) #tokenize sentences
    
    #File = open('input.txt') #open file
    lines_common = file_name #read all lines
    stop = stopwords.words('english')  #remove stopwords
    lines_common=[i for i in word_tokenize(lines_common.lower()) if i not in stop]
     #calculate most freq word length
    allWordDist = nltk.FreqDist(w.lower() for w in lines_common)
    common = [len(i[0]) for i in allWordDist.most_common(1)]
    common_word =[i[0] for i in allWordDist.most_common(1)]
    print("most common word length")
    print (common)
    average = sum(len(word) for word in lines_common) / len(lines_common) 
    print("average word length")
    print (average)

    s_adj = [] #empty to array to hold all superlative adj
    c_adj=[] #empty arrayto hold comparative adjectives
    pdt = [] #empty array to hold predeterminants
    c_conj = [] #empty array to hold coordinating conjunctions

    ex_there = [] #empty array to hold exesntial there
    s_adv =[] #empty array to hold superlative adverb
    for sentence in sentences:
         for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
             if (pos == 'JJS'):
                 s_adj.append(word)
             elif (pos == 'PDT'):
                 pdt.append(word)
             elif (pos == 'CC'):
                 c_conj.append(word)
             elif (pos=='EX'):
                 ex_there.append(word)
             elif(pos=='JJR'):
                 c_adj.append(word)
             elif(pos=='RBS'):
                s_adv.append(word)
    ex_there_length = len(ex_there)
    s_adj_length = len(s_adj)
    pdt_length = len(pdt)
    c_conj_length = len(c_conj)
    c_adj_length = len(c_adj)
    s_adv_length=len(s_adv)
    print 'number of superlative adj'
    print (s_adj_length)
    print 'number of predeterminants'
    print (pdt_length)
    print 'number of coordinating conjuctions'
    print (c_conj_length)
    print 'number of exesntial there'
    print (ex_there_length)
    print 'number of comparative adjectives'
    print (c_adj_length)
    print 'number of superlative adverb'
    print(s_adv_length)
    
    return pos_unique,count,ex_there_length,s_adj_length,pdt_length,c_conj_length,c_adj_length,s_adv_length,words_len,characters,common,average

# print "syntax properties of essay:" + str(syntax_results('input.txt'))

def get_results(file_name):
    # text1 = open(file_name,'r')#reading essay from input
    text_sents = sent_tokenize(file_name) #tokenize the sentence

    download_dir = "results_essay_code_magnitude.csv" #where you want the file to be downloaded to 
    csv = open(download_dir, "w") 
    columnTitleRow = "s1, s2,output\n"
    csv.write(columnTitleRow)
    results = [] #creating an array to save results grater than 0.6
    results_all =[]  #creating an array to save all results 
    for i in range(len(text_sents)):
      #print i
      results.append([])
      results_all.append([])
      #j=i+1
      for j in range(len(text_sents)):
            
       s1=text_sents[i] 
       s1=s1.decode('latin-1').encode('utf8')
       s2=text_sents[j]
       s2=s2.decode('latin-1').encode('utf8')
       vector_1 = form_value_vector(s1, s2)
      # print "V1",vector_1
       vector_2 = form_value_vector(s2, s1)
       #print "V2",vector_2
       #print lines[1].rstrip()
       output=similarity(vector_1, vector_2, s1, s2)
       if (output==1.0): #append 0 in both the arrays if output =1 
          results[i].append(0)
          results_all[i].append(0) 
       else:
         if(output > 0.1):#append output to both the arrays if it is greater tahn 0.6
            results[i].append(output)
            results_all[i].append(output)
         else: #append output to only the result_all array if grather than 0.6 and 0 if less
            results_all[i].append(output)
            results[i].append(0)  
         

       #print output
            #print "\n"
       dic = {s1:s2}
       for key in dic.keys():
               s1 = key
               s2 = dic[key]
               row = "'(" + str(i) + ") " + s1 +  "','"  + "(" + str(j) + ") "  + s2 +  "','" + str(output) + "'\n"
               csv.write(row)

    return results,results_all

# results,results_all=get_results('input.txt')
import pickle
import os.path


def compare_topwords(l1,l2):
  
   vector_single  = form_value_vector_tfidf(l1, l2) #just single vector beacuse have to compute just for words
      # print "V1",vector_1
  
       #print "V2",vector_2
       #print lines[1].rstrip()
   output=similarity_tfidf(vector_single, l1, l2)
   return output
# results = None 
# results_all= None

# # Caching result
# if not (os.path.isfile('cache_result') and os.path.isfile('cache_result_all')):    
#     results,results_all=get_results('input.txt')
#     with open('cache_result', 'wb') as fp:
#         pickle.dump(results, fp)
#     with open('cache_result_all', 'wb') as fp:
#         pickle.dump(results_all, fp)
# else:
#     with open('cache_result', 'rb') as fp:
#         results = pickle.load(fp)
#     with open('cache_result_all', 'rb') as fp:
#         results_all = pickle.load(fp)



import matplotlib.pyplot as plt
import networkx as nx
import numpy


def negate_matrix(result, factor=1):
    import copy
    _result = copy.deepcopy(result)
    for x in range(0,len(result)):
        for y in range(0, len(result[x])):
            _result[x][y] = result[x][y] * factor * -1
    return _result

#GENerating a graph of negation matrix to be used for closeness centrality and others with filtered results for >0.6
# A_neg = numpy.matrix(negate_matrix(results, factor=1))
# G_neg = nx.from_numpy_matrix(A_neg)

# #GENerating a graph of negation matrix to be used for closeness centrality and others with all results {will use these in some attributes}
# A_neg_all = numpy.matrix(negate_matrix(results_all, factor=1))
# G_neg_all = nx.from_numpy_matrix(A_neg_all)

def generate_graph(result, view = 0):
    plt.figure(view)
    A = numpy.matrix(result)
    G = nx.from_numpy_matrix(A)
    pos=nx.spring_layout(G)
    nx.draw(G, pos, labels={i: str(i) for i in range(0,len(result))}, font_size=12, weight='length')
    #nx.draw_networkx_edge_labels(G, pos,)
    plt.show()


# Threshold for semantic similarity is > 0.6




# Distance between sentences = distance between nodes
# generate_graph(results, view=1)



# # Now multiplying by -1 to make
# # distance between sentences inverse of distance between nodes
# generate_graph(negate_matrix(results, factor=100), view=2)


# Now creating Minimum Spanning Tree
# mst = minimum_spanning_tree(csr_matrix(results)).toarray().astype(float)
# generate_graph(mst, view=3)


def mst_sum(_mst):
    s = 0
    for x in _mst:
        for y in x:
            s = s + y
    return s

# print "MST sum: "  + str(mst_sum(list(mst)))



# # Now creating Maximum Spanning Tree using Negation of Matrix
# max_st = minimum_spanning_tree(csr_matrix(negate_matrix(results,100))).toarray().astype(float)
# # generate_graph(max_st, view=4)


def max_st_sum(_max_st):
    s=0
    for x in _max_st:
        for y in x:
            s=s+y
    s=(s/100) * -1
    return s  

# print "Maximum ST sum: " +str(max_st_sum(list(max_st)))

def closeness_centrality(G, u=None, distance=None, normalized=True): #using g_negative #higher value indicated higer centrality
    
    if distance is not None:
        # use Dijkstra's algorithm with specified attribute as edge weight 
        path_length = functools.partial(nx.single_source_dijkstra_path_length,
                                        weight=distance)
    else:
        path_length = nx.single_source_shortest_path_length

    if u is None:
        nodes = G.nodes()
    else:
        nodes = [u]
    closeness_centrality = {}
    for n in nodes:
        sp = path_length(G,n)
        totsp = sum(sp.values())
        if totsp > 0.0 and len(G) > 1:
            closeness_centrality[n] = (len(sp)-1.0) / totsp
            # normalize to number of nodes-1 in connected part
            if normalized:
                s = (len(sp)-1.0) / ( len(G) - 1 )
                closeness_centrality[n] *= s
        else:
            closeness_centrality[n] = 0.0
    cc_sum=0        
    if u is not None:
        return closeness_centrality[u]
    else:
        for i in closeness_centrality:
            cc_sum=cc_sum+closeness_centrality[i]

            cc_avg=cc_sum/len(closeness_centrality)
            return cc_avg
            

# print "closeness_centrality:" + str(closeness_centrality(G_neg, u=None, distance=None, normalized=True)) 


#center
#diameter
#density
#dispersion
#diff in number of nodes in results and number of nodes in results_all
__all__ = ['eccentricity', 'diameter', 'radius', 'periphery', 'center','eigenvector_centrality']

def eccentricity(G, v=None, sp=None):
    order=G.order()

    e={}
    for n in G.nbunch_iter(v):
        if sp is None:
            length=nx.single_source_shortest_path_length(G,n)
            L = len(length)
        else:
            #try:
            length=sp[n]
            L = len(length)
            # except TypeError:
                # raise nx.NetworkXError('Format of "sp" is invalid.')
        if L != order:
             # msg = "Graph not connected: "
             # raise nx.NetworkXError(msg)
             e[n]= float("inf")
           # L = float("inf")
        else:
              e[n]=max(length.values())
    e_sum=0
    if v in G:
        return e[v]  # return single value
    else:
        return e


# print "eccentricity of nodes:" +str(eccentricity(G_neg,v=None, sp=None)) #with filtered results or lese it gives 1

def diameter(G, e=None):
    if e is None:
        e=eccentricity(G)
    return max(e.values())

# print "diameter:" + str(diameter(G_neg,e=None)) #with all results to get the correct diameter

def number_of_nodes(G):
    """Return the number of nodes in the graph."""
    return G.number_of_nodes()

def number_of_edges(G):
    """Return the number of edges in the graph. """
    return G.number_of_edges()

def density(G):
  
  n=number_of_nodes(G)
  m=number_of_edges(G)
  if m==0 or n <= 1:
        d=0.0
  else:
        if G.is_directed():
            d=m/float(n*(n-1))
        else:
            d= m*2.0/float(n*(n-1))
  return d
# final_density = (density(G_neg_all)) - (density(G_neg))

# print "difference in the density of graphs with only strongly similar sentneces and all the sentences:" + str(final_density) #less is diff the better it is


def radius(G, e=None):
    # "Return the radius of the graph G.

    # The radius is the minimum eccentricity.
   
    if e is None:
        e=eccentricity(G)
    return min(e.values())

# print "radius of the graph is:" +str(radius(G_neg,e=None))

def center(G, e=None):
    # ""Return the center of the graph G. 

    # The center is the set of nodes with eccentricity equal to radius. 

    if e is None:
        e=eccentricity(G)
    # order the nodes by path length
    radius=min(e.values())
    p=[v for v in e if e[v]==radius]
    return len(p)
# print "number of nodes in center of the graph is:" +str(center(G_neg,e=None))

__all__ = ['dispersion']



def dispersion(G, u=None, v=None, normalized=True, alpha=1.0, b=0.0, c=0.0):


    def _dispersion(G_u, u, v):
            """dispersion for all nodes 'v' in a ego network G_u of node 'u'"""
            u_nbrs = set(G_u[u])
            ST = set(n for n in G_u[v] if n in u_nbrs)
            set_uv = set([u, v])
            # all possible ties of connections that u and b share
            possib = combinations(ST, 2)
            total = 0
            for (s, t) in possib:
                # neighbors of s that are in G_u, not including u and v
                nbrs_s = u_nbrs.intersection(G_u[s]) - set_uv
                # s and t are not directly connected
                if t not in nbrs_s:
                    # s and t do not share a connection
                    if nbrs_s.isdisjoint(G_u[t]):
                        # tick for disp(u, v)
                        total += 1
            # neighbors that u and v share
            embededness = len(ST)

            if normalized:
                if embededness + c != 0:
                    norm_disp = ((total + b)**alpha) / (embededness + c)
                else:
                    norm_disp = (total + b)**alpha
                dispersion = norm_disp

            else:
                dispersion = total

            return dispersion

    if u is None:
        # v and u are not specified
        if v is None:
            results = dict((n, {}) for n in G)
            for u in G:
                for v in G[u]:
                    results[u][v] = _dispersion(G, u, v)
        # u is not specified, but v is
        else:
            results = dict.fromkeys(G[v], {})
            for u in G[v]:
                results[u] = _dispersion(G, v, u)
    else:
        # u is specified with no target v
        if v is None:
            results = dict.fromkeys(G[u], {})
            for v in G[u]:
                results[v] = _dispersion(G, u, v)
        # both u and v are specified
        else:
            results = _dispersion(G, u, v)
            # results_len=len(results)
    sum=0        
    # length=len(results)
    for the_key, the_value in results.iteritems():
     for i in the_value:
       # print the_value[i]
       sum=sum+the_value[i]
    return sum   


# print "center of the graph is:" +str(dispersion(G_neg, u=None, v=None, normalized=True, alpha=1.0, b=0.0, c=0.0))

#calling syntax properties of graph

import math
from textblob import TextBlob as tb

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

def tfidf_main(document1):


    bloblist = [document1]
    
    top_words=[]
    for i, blob in enumerate(bloblist):
        
        scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        sorted_words_filter=[]
        
        for word, score in sorted_words[:10]:

          
          pos= nltk.pos_tag(word)
          
          if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' or pos=='VBP' or pos=='VB',pos=='VBD',pos=='VBG',pos=='VBN',pos=='VBZ'):
              sorted_words_filter.append(word)
              top_words.append(word)
              
    return top_words





def main(input_data):


    file =input_data
    document1 = tb("""
        More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. 
        Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you
   """)
    top_words_question=[]
    top_words_question=tfidf_main(document1)

    essays = pd.read_csv(file)
    download_dir = "/home/compute/work/aee/essay_evaluation_codes/domain1/0.1/all_0.1.csv" #where you want the file to be downloaded to 
    
    _csv = open(download_dir, "w")
    
    _csv_file = open(download_dir, "w") 
    # Headers for CSV
    headers = ['Essay','pos_unique','misspelled words','coordinating_conjuctions',
    'words','characters','min_st_sum','max_st_sum','c_centrality',
    'density_diff','top_words_comp','common_length']
    writer = csv.DictWriter(_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, fieldnames=headers, extrasaction='ignore')
    writer.writeheader()
    # Counter variable to know essay index
    essay_counter = 1

    for index, row in essays.iterrows():
        s1=row['essay']
        s1=s1.decode("utf-8", "ignore")
        s1=s1.encode('ascii', 'ignore')
        #results=get_results(s1)
        #negate_matrix(results, factor=1)
        top_words_essay=[]
        s1_tfidf=tb(s1)
        top_words_essay=tfidf_main(s1_tfidf)
        comp_results=compare_topwords(top_words_question,top_words_essay)
        print 'top words diff'
        print comp_results
        pos_unique,count,ex_there_length,s_adj_length,pdt_length,c_conj_length,c_adj_length,s_adv_length,words_len,characters,common,average=syntax_results(s1)
        
        results,results_all=get_results(s1)
        
        # generate_graph(results_all, view = 0)
        # generate_graph(results, view = 0)

        negate_results=negate_matrix(results, factor=1)
       
        # generate_graph(negate_results, view = 0)
         
        mst = minimum_spanning_tree(csr_matrix(results_all)).toarray().astype(float)

        # generate_graph(mst, view=3)

        mst_sum1 = str(mst_sum(list(mst)))
        
        max_st = minimum_spanning_tree(csr_matrix(negate_matrix(results_all,100))).toarray().astype(float)
        # generate_graph(max_st, view=3)
        max_st_sum1 = str(max_st_sum(list(max_st)))
         
        A_neg_all = numpy.matrix(negate_matrix(results_all, factor=1))
        G_neg_all = nx.from_numpy_matrix(A_neg_all)



        A_neg = numpy.matrix(negate_matrix(results, factor=1))
        G_neg = nx.from_numpy_matrix(A_neg)
    
        c_centrality=str(closeness_centrality(G_neg, u=None, distance=None, normalized=True)) 
         
        # diameter1 =str(diameter(G_neg,e=None))

        final_density = (density(G_neg_all)) - (density(G_neg))
        
        # radius1=str(radius(G_neg,e=None))
        
        # center1= str(center(G_neg_all,e=None))
        
        # dispersion1 = str(dispersion(G_neg, u=None, v=None, normalized=True, alpha=1.0, b=0.0, c=0.0))
        
        # eigen = str(eigen1(G_neg_all))



        output = {'Essay': s1,'pos_unique': str(pos_unique),'misspelled words': str(count),
        'coordinating_conjuctions':str(c_conj_length),
        'words':str(words_len),'characters':str(characters),'common_length':str(common),
        'min_st_sum':mst_sum1,'max_st_sum':max_st_sum1,
        'c_centrality':c_centrality,
        'density_diff':final_density,
        'top_words_comp':comp_results}
        writer.writerow(output)
        essay_counter = essay_counter + 1
        print "essay number"
        print essay_counter
        
      

     #pos_unique,count,ex_there_length,s_adj_length,pdt_length,c_conj_length,c_adj_length,s_adv_length,words,characters=syntax_results(s1)
     # dic = {essay}
     # for key in dic.keys():
     #     s1 = key
     #     s2 = dic[key]
     #     row = "\"" +essay +  "\"" +  "," + "\""   +pos_unique  +  "\"" + "," +  "\"" +ex_there_length  +  "\"" + "\n"
     #     csv.write(row)
                



main('/home/compute/work/aee/essay_evaluation_codes/domain1/results_attri_domain1.csv')

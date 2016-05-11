
#from bs4 import BeautifulSoup
#import urllib

#r = urllib.urlopen('http://www.imdb.com/title/tt2776304/?ref_=nm_flmg_act_12').read()
#soup = BeautifulSoup(r, 'lxml')
#div = soup.find_all("div", class_="summary_text")

# In Matthew's environment, a harmless exception is thrown from the following import.
# Actually, it seems to be the first import, whatever it is.  Anyway, just ignore it.
try:
    from nltk.tokenize import wordpunct_tokenize
except Exception:
    pass

import numpy as np
import codecs
import nltk
import re
import csv
import json
import sys
import operator
from nltk import PorterStemmer
from math import log
from collections import Counter

################################################################################
class Corpus():
    """
    The Corpus class represents a document collection.
    """
    def __init__(self, doc_data, stopword_file, clean_length):
        """
        Notice that the __init__ method is invoked everytime an object of the
        class is instantiated.
        """
        # Initialise documents by invoking the appropriate class
        self.docs = [Document(doc[0], doc[1], doc[2]) for doc in doc_data]         
        self.N = len(self.docs)
        self.clean_length = clean_length
        
        # Get a list of stopwords
        self.create_stopwords(stopword_file, clean_length)
        
        # Stopword removal, token cleaning and stemming to docs
        self.clean_docs(2)
        
        # Create vocabulary
        self.corpus_tokens()
        
    def clean_docs(self, length):
        """ 
        Applies stopword removal, token cleaning and stemming to docs.
        """
        for doc in self.docs:
            doc.token_clean(length)
            doc.stopword_remove(self.stopwords)
            doc.stem()        
    
    def create_stopwords(self, stopword_file, length):
        """
        Description: parses a file of stopwords, removes words of length
        'length' and  stems it.
        input: length: cutoff length for words
               stopword_file: stopwords file to parse
        """        
        with codecs.open(stopword_file, 'r', 'utf-8') as f: raw = f.read()        
        self.stopwords = (np.array([PorterStemmer().stem(word) 
                                    for word in list(raw.splitlines()) if len(word) > length]))
             
    def corpus_tokens(self):
        """
        Description: create a set of all all tokens or in other words a
        vocabulary
        """        
        # Initialise an empty set
        self.token_set = set()
        for doc in self.docs:
            self.token_set = self.token_set.union(doc.tokens) 

################################################################################
class Document():    
    """
    The Doc class represents a class of individual documents
    """    
    def __init__(self, speech_year, speech_pres, speech_text):
        self.year = int(speech_year)
        self.pres = speech_pres
        self.text = speech_text.lower()
        self.tokens = np.array(wordpunct_tokenize(self.text))
    
    def friendly_string(self):
        """ 
        Description: generate a friendly string to describe the document
        """
        return "{0} {1} {2}".format(self.year, self.pres, self.text[1:20])
        
    def token_clean(self,length):
        """ 
        Description: strip out non-alpha tokens and tokens of length > 'length'
        input: length: cut off length 
        """
        self.tokens = np.array([t for t in self.tokens if
                                (t.isalpha() and len(t) > length)])

    def stopword_remove(self, stopwords):
        """
        Description: remove stopwords from tokens.
        input: stopwords: a suitable list of stopwords
        """
        self.tokens = np.array([t for t in self.tokens if t not in stopwords])

    def stem(self):
        """
        Description: stem tokens with Porter Stemmer.
        """
        self.tokens = np.array([PorterStemmer().stem(t) for t in self.tokens])

    def term_vector(self, corpus_token_list):
        """
        Description: generate a term-vector for this document.  The result
                     corresponds with a single row of the document-term-matrix
                     of the corpus
        input: corpus_token_list: a list of tokens from the corpus, a subset
                                  of which will be found in this document.
        """
        vector = [None] * len(corpus_token_list)
        counter = Counter(self.tokens)
        for i in range(len(corpus_token_list)):
            count = counter[corpus_token_list[i]]
            vector[i] = count

        return vector

################################################################################
class Driver:
    """
    The Driver class oversees the TF-IDF/SVD analysis that we need to do.
    It collects the functions into a clean container.
    """
    def __init__(self, text_path, stopwords_path):
        """
        Initialize our driver object, loading the corpus and calculating
        its TF-IDF matrix.
        """
        text = open(text_path, 'r').read()
        regex = '_(\d{4}).*?_[a-zA-Z]+.*?_[a-zA-Z]+.*?_([a-zA-Z]+)_\*+(\\n{2}.*?)\\n{3}'
        pres_speech_list = self.parse_text(text, regex)
        self.corpus = Corpus(pres_speech_list, stopwords_path, 2)

    def parse_text(self, textraw, regex):
        """
        Takes raw string and performs two operations:
          1. Breaks text into a list of speech, president and speech
          2. Breaks speech into paragraphs
        """
        prs_yr_spch_reg = re.compile(regex, re.MULTILINE | re.DOTALL)
    
        # Each tuple contains the year, name of the president and the speech
        # text
        prs_yr_spch = prs_yr_spch_reg.findall(textraw)
    
        # Convert immutabe tuple to mutable list
        prs_yr_spch = [list(tup) for tup in prs_yr_spch]
        for i in range(len(prs_yr_spch)):
            prs_yr_spch[i][2] = prs_yr_spch[i][2].replace('\n', '')
    
        # Sort
        prs_yr_spch.sort()
    
        return prs_yr_spch

    """
    Use a Gibbs sampler for Latent Dirichilet Allocation.  Write the results
    to an output file.
    """
    def Gibbs_LDA(self, K, eta, theta, max_iterations, convergence_cutoff, output_path):

        f = open(output_path, 'w')

        D = len(self.corpus.docs) # number of documents

        vocab = list(self.corpus.token_set)
        V = len(vocab) # number of words in vocabulary

        # Initialize hyperparameters according to recommendations of
        # Griffiths and Steyvers.
        eta_vector = np.full(shape = V, fill_value = eta)
        alpha_vector = np.full(shape = K, fill_value = theta)

        # Initialize an empty matrix for the m and n updating parameters
        # so that they have no impact on the first iteration.
        m = np.zeros(shape = (K, V))
        n = np.zeros(shape = (D, K))
        B_norm_prev = None
        T_norm_prev = None

        for iteration in range(max_iterations):
            
            # Sample from a Dirichlet K times with the current hyperparameter
            # eta.
            B = np.zeros(shape = (V, K))
            for k in range(K):
                B[:, k] = np.random.dirichlet(m[k, :] + eta_vector)

            # Sample from a Dirichlet D times with the current hyperparameter
            # alpha
            T = np.zeros(shape = (K, D))
            for d in range(D):
                T[:, d] = np.random.dirichlet(n[d, :] + alpha_vector)

            # Sample a topic z for each word of each document from a
            # multinomial with the current values of beta (B)
            # and theta (T).
            z = np.zeros(shape = (D, V))
            for d in range(D):
                for v in range(V):
                    numer = np.zeros(K)
                    denom = 0
                    for k in range(K):
                        numer[k] = T[k, d] * B[v, k]
                        denom += numer[k]
                    values = numer / denom
                    z[d, v] = np.random.multinomial(1, values).argmax()

            # Initialize the new updating parameters m and n.
            m = np.zeros(shape=(K, V))
            n = np.zeros(shape=(D, K))

            # Compute current updating parameters m and n using the current
            # draws of topics per word.
            for d in range(D):
                doc = self.corpus.docs[d]
                for word in doc.tokens:
                    v = vocab.index(word)
                    k = z[d, v]
                    m[k, v] += 1
                    n[d, k] += 1

            # Compute the Frobenius norm for each of the matrices beta (B)
            # and theta (T).
            B_norm = np.linalg.norm(B)
            T_norm = np.linalg.norm(T)

            # Check if the improvements are both smaller than the
            # convergence cutoff.  If so, exit the loop.
            if (B_norm_prev != None):
                if (abs(B_norm - B_norm_prev) < convergence_cutoff
                    and abs(T_norm - T_norm_prev) < convergence_cutoff):
                    break

            # Save off the values of the B and T norms for comparison in the
            # next iteration.
            B_norm_prev = B_norm
            T_norm_prev = T_norm

        # Print a subset of the 10 most probable words for each of the topics
        for k in range(K):
            sorted_word_indexes = sorted(range(V), key=lambda x:B[x, k])
            top_word_indexes = sorted_word_indexes[-10:]
            f.writelines("-------------------------\n")
            f.writelines("Topic {0}\n".format(k))
            for word in top_word_indexes:
                f.writelines("    Word: {0} ({1})\n".format(vocab[word], B[word, k]))

        # Print a subset of the 3 most probable topics for each document
        for d in range(D):
            sorted_topic_indexes = sorted(range(K), key=lambda x:T[x, d])
            top_topic_indexes = sorted_topic_indexes[-3:]
            f.writelines("-------------------------\n")
            f.writelines("Document {0}, {1} {2}\n".format(d, self.corpus.docs[d].pres, self.corpus.docs[d].year))
            for topic in top_topic_indexes:
                f.writelines("    Topic {0} ({1})\n".format(topic, B[d, topic]))

        f.close()

# Create the driver object to oversee the process.
driver = Driver('../data/pres_speech/sou_all.txt',
                 '../data/stopwords/stopwords.txt')

V = len(driver.corpus.token_set) # the number of words in the vocabulary
K = 5 # the number of topics

driver.Gibbs_LDA(K, eta=200.0/V, theta=50.0/K,
                 max_iterations=20, convergence_cutoff=0.01,
                 output_path="../data/output.txt")

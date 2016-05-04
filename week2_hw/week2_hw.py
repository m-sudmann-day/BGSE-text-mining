################################################################################
# TF-IDF and using Singular Value Decomposition and Cosine Similarity to
# identify clusters of documents.
################################################################################

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
    
    def document_term_matrix(self):
        """
        Description: generate the document-term matrix for the corpus
        """        
        result = []
        for doc in self.docs:
            vector = doc.term_vector(list(self.token_set))
            result.append(vector)        
        
        return result

    def tf_idf(self):
        """
        Description: generate the TF-IDF matrix for this corpus
        """

        # Generate a copy of the document-term matrix to work with in this
        # function and initialize other local variables.
        dt_matrix = self.document_term_matrix()
        tf_matrix = []
        idf_matrix = []
        tf_idf_matrix = []

        # Build a term frequency matrix from the document term matrix.
        # tf(d,v) = { 0 if x(d,v) = 0, 1 + log(x(d), v) otherwise }
        for dt_doc_vector in dt_matrix:
            tf_doc_vector = [(0 if x == 0 else 1 + log(x)) for x in dt_doc_vector]
            tf_matrix.append(tf_doc_vector)

        # Build a document frequency matrix for each term.
        # Initialize with zeros.
        df_vector = np.zeros(len(self.token_set))
        for dt_doc_vector in dt_matrix:
            # Increment the counters based on an indicator function which
            # is 1 if there is at least one instance of the term in the doc.
            df_vector = np.add(df_vector, [int(x > 0) for x in dt_doc_vector])

        # Build an inverse document frequency vector.
        idf_doc_vector = [log(len(self.docs) / x) for x in df_vector]

        # Build the TF-IDF weighting matrix.
        for tf_doc_vector in tf_matrix:
            tf_idf_vector = np.multiply(tf_doc_vector, idf_doc_vector)
            tf_idf_matrix.append(tf_idf_vector)

        return tf_idf_matrix

    def dict_rank(self, dictionary, use_tf_idf, n):        
        """
        Description: rank the documents in this corpus against the provided
        dictionary.  Return the top n documents.
        input: dictionary: the dictionary against which to rank the documents
               use_tf_idf: True if the TF-IDF matrix is to be used; False if
                           the document-term matrix is to be used.
               n: the number of top-ranked documents to return
        """
        if (use_tf_idf):
            dtm = self.tf_idf()
        else:
            dtm = self.document_term_matrix()
            
        # Get rid of words in the document term matrix not in the dictionary
        dict_tokens_set = set(item for item in dictionary)
        intersection = dict_tokens_set & self.token_set
        vec_positions = [int(token in intersection) for token in self.token_set] 

        # Get the score of each document
        sums = np.zeros(len(dtm))
        for j in range(len(dtm)):
            sums[j] = sum([a * b for a, b in zip(dtm[j], vec_positions)])

        # Order them and return the n top documents
        order = sorted(range(len(sums)), key = lambda k: sums[k], reverse=True)
        ordered_doc_data_n = [None] * len(dtm)
        ordered_sums = np.zeros(len(dtm))

        counter = 0        
        for num in order:
            ordered_doc_data_n[counter] = self.docs[num]
            ordered_sums[counter] = sums[num]
            counter += 1

        return zip(ordered_doc_data_n[0:n], ordered_sums[0:n])

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
        self.tf_idf = self.corpus.tf_idf()
        self.init_csv_file()

    def parse_text(self, textraw, regex):
        """
        Takes raw string and performs two operations:
          1. Breaks text into a list of speech, president and speech
          2. Breaks speech into paragraphs
        """
        prs_yr_spch_reg = re.compile(regex, re.MULTILINE|re.DOTALL)
    
        # Each tuple contains the year, name of the president and the speech text
        prs_yr_spch = prs_yr_spch_reg.findall(textraw)
    
        # Convert immutabe tuple to mutable list
        prs_yr_spch = [list(tup) for tup in prs_yr_spch]
        for i in range(len(prs_yr_spch)):
            prs_yr_spch[i][2] = prs_yr_spch[i][2].replace('\n', '')
    
        # Sort
        prs_yr_spch.sort()
    
        return prs_yr_spch

    def cosine_similarity(self, v1, v2):
        """
        Calculate the cosine similarity of two vectors (vectors of terms
        in a document).
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        return np.dot(v1, v2) / (norm1 * norm2)

    def compare_cosine_similarities(self, cos_sim_matrix, indexes1, indexes2):
        """
        Calculate the average cosine similarities of a subset of documents
        in the corpus as defined by their indexes.
        """
        total = 0
        count = 0
        for i in indexes1:
            for j in indexes2:
                if i != j:
                    total += cos_sim_matrix[i, j]
                    count += 1
        if (count == 0):
            return 0
        return total/count

    def init_csv_file(self):
        """
        Create the output CSV file and write the header row.
        """
        f = open('output.csv', 'w')
        f.writelines("analysis_num,max_singular_values,AA,AB,BB\n")
        f.close()

    def analyze(self, analysis_num, lambda1, lambda2):
        """
        Perform an analysis in which two subsets of the documents
        are compared to each other.  They may or may not overlap.
        These subsets are identified by the lambda express applied
        to the full set of documents in the corpus.
        Results are average cosine similarities by number of retained
        singular values and are writing to the output CSV.
        """
        # Generate X2 which is the reconstituted "X hat" in the SVD
        # case, and simply a copy of X in the normal case.
        N = len(self.corpus.docs)

        # Obtian document subsets by lambda expression.
        docs1 = [doc for doc in self.corpus.docs if lambda1(doc)]
        docs2 = [doc for doc in self.corpus.docs if lambda2(doc)]

        # Convert the lists of documents to lists of indexes.
        indexes1 = [self.corpus.docs.index(doc) for doc in docs1]
        indexes2 = [self.corpus.docs.index(doc) for doc in docs2]

        # Loop, skipping to each 10th singular value.
        for max_singular_values_10 in range(1+N/10):

            max_singular_values = max_singular_values_10 * 10

            if (max_singular_values > 0):

                # Decompose
                U, s, V = np.linalg.svd(self.tf_idf)

                # Chop off some of the singular values.
                if (max_singular_values < len(s)):
                    s = np.append(s[:max_singular_values],
                                  np.zeros(len(s) - max_singular_values))

                S = np.zeros((len(U), len(V)))

                S[:len(s), :len(s)] = np.diag(s)

                # X is really X_hat in this case.
                X = np.array(np.dot(U, np.dot(S, V)))

            else:

                X = np.array(self.tf_idf)

            cos_sim_matrix = np.zeros((N, N))

            # Calculate cosine similarities for all possible pairs of documents.
            for i in range(N):
                for j in range(N):
                    cos_sim_matrix[i,j] = self.cosine_similarity(X[i, :], X[j, :])

            # Calculate average cosine similarities within and between the two subsets.
            avg_sim11 = self.compare_cosine_similarities(cos_sim_matrix,
                                                         indexes1, indexes1)
            avg_sim12 = self.compare_cosine_similarities(cos_sim_matrix,
                                                         indexes1, indexes2)
            avg_sim22 = self.compare_cosine_similarities(cos_sim_matrix,
                                                         indexes2, indexes2)

            # Write to the output CSV.
            f = open('output.csv', 'a')
            s = "{0},{1},{2},{3},{4}\n".format(analysis_num, max_singular_values,
                                 avg_sim11, avg_sim12, avg_sim22)
            f.writelines(s)
            f.close()

            print max_singular_values, avg_sim11, avg_sim12, avg_sim22

# Create the driver object to oversee the process.
driver = Driver ('./../data/pres_speech/sou_all.txt',
                 './../data/stopwords/stopwords.txt')

# Create lists of Democrats and Republicans since 1900. The Roosevelts are
# handled later.
definitely_dem = ['Obama', 'Carter', 'Johnson', 'Kennedy', 'Truman', 'Wilson']
definitely_rep = ['Bush', 'Reagan', 'Ford', 'Nixon', 'Eisenhower',
                  'Hoover', 'Coolidge', 'Harding', 'Taft', 'McKinley']

# Set up lambda expressions for extracting subsets of the corpus.
lambda_obama = lambda doc : doc.pres == "Obama"
lambda_wbush = lambda doc : doc.pres == "Bush" and doc.year > 1999
lambda_dem_since_1900 = lambda doc : doc.pres in definitely_dem or (doc.pres == "Roosevelt" and doc.year > 1920)
lambda_rep_since_1900 = lambda doc : doc.pres in definitely_rep or (doc.pres == "Roosevelt" and doc.year < 1920)
lambda_century_19 = lambda doc : (doc.year >= 1800) and (doc.year < 1900)
lambda_century_20 = lambda doc : (doc.year >= 1900) and (doc.year < 2000)

# Run the three analyses.
driver.analyze(1, lambda_obama, lambda_wbush)
driver.analyze(2, lambda_dem_since_1900, lambda_rep_since_1900)
driver.analyze(3, lambda_century_19, lambda_century_20)

# The output now exists in the output CSV.
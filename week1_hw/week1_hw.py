import numpy as np

# In my environment, a harmless exception is thrown from the following import.
# Just ignore it.
try:
    from nltk.tokenize import wordpunct_tokenize
except Exception:
    pass

import codecs
import nltk
import re
from nltk.tokenize import wordpunct_tokenize
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
        description: parses a file of stowords, removes words of length 'length'
        and  stems it.
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
        result = []
        for doc in self.docs:
            vector = doc.term_vector(list(self.token_set))
            result.append(vector)        
        
        return result

    def tf_idf(self):

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
        df_vector = [0] * len(self.token_set)
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

    def dict_rank(self, dictionary, n):        
        dtm = self.document_term_matrix()
        all_tokens = list(self.token_set)
        
        # Get rid of words in the document term matrix not in the dictionary
        vec_positions = [0] * len(dtm[0])        
        for i in range(len(all_tokens)):
            if all_tokens[i] in dictionary:
                vec_positions[i] = 1
            else:
                vec_positions[i] = 0
        sums = [0] * len(dtm)

        # Get the score of each document
        for j in range(len(dtm)):
            sums[j] = sum([a * b for a, b in zip(dtm[j], vec_positions)])

        # Order them and return the n top documents
        order = sorted(range(len(sums)), key = lambda k: sums[k])
        ordered_doc_data_n = [0] * len(dtm)
        counter = 0        
        for num in order:
            ordered_doc_data_n[counter] = doc_data[num]
            counter += 1
        n_top = ordered_doc_data_n[0:n]
       
        return n_top

################################################################################
class Document():    
    """
    The Doc class represents a class of individual documents
    """    
    def __init__(self, speech_year, speech_pres, speech_text):
        self.year = speech_year
        self.pres = speech_pres
        self.text = speech_text.lower()
        self.tokens = np.array(wordpunct_tokenize(self.text))
        
    def token_clean(self,length):
        """ 
        Description: strip out non-alpha tokens and tokens of length > 'length'
        input: length: cut off length 
        """
        self.tokens = np.array([t for t in self.tokens if (t.isalpha() and len(t) > length)])

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

    def term_vector(self, doc_token_list):
        vector = [None] * len(doc_token_list)
        counter = Counter(self.tokens)
        for i in range(len(doc_token_list)):
            count = counter[doc_token_list[i]]
            vector[i] = count

        return vector

################################################################################
def parse_text(textraw, regex):
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

################################################################################
text = open('./../data/pres_speech/sou_all.txt', 'r').read()
regex = '_(\d{4}).*?_[a-zA-Z]+.*?_[a-zA-Z]+.*?_([a-zA-Z]+)_\*+(\\n{2}.*?)\\n{3}'
pres_speech_list = parse_text(text, regex)

# Instantiate the corpus class
corpus = Corpus(pres_speech_list, './../data/stopwords/stopwords.txt', 2)
tf_idf = corpus.tf_idf()

print corpus
print corpus.docs[0]
print(tf_idf)
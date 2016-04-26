
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
        self.year = speech_year
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

#################################################################################
text = open('./../data/pres_speech/sou_all.txt', 'r').read()
regex = '_(\d{4}).*?_[a-zA-Z]+.*?_[a-zA-Z]+.*?_([a-zA-Z]+)_\*+(\\n{2}.*?)\\n{3}'
pres_speech_list = parse_text(text, regex)

# Instantiate the corpus class
corpus = Corpus(pres_speech_list, './../data/stopwords/stopwords.txt', 2)
tf_idf = corpus.tf_idf()

#################################################################################
# Harvard IV set
file_handler = './../data/dictionary/inquirerbasic2.csv'

dictionary = np.loadtxt(open(file_handler, 'rb'), dtype = 'str',
                        delimiter = ';', skiprows = 1, comments = None)

our_dictionary = sorted(set(elem[0].rstrip('#01234256789').lower() for elem in dictionary))

n = 10

# Document term matrix

scored_docs = corpus.dict_rank(our_dictionary, False, n)
print "The highest ranked documents using DTM are:"
for i in range(len(scored_docs)):
    print "{0} {1} {2}".format(scored_docs[i][0].year, scored_docs[i][0].pres, scored_docs[i][1])

# TF-IDF
scored_docs = corpus.dict_rank(our_dictionary, True, n)
print "The highest ranked documents using TF-IDF are:"
for i in range(len(scored_docs)):
    print "{0} {1} {2}".format(scored_docs[i][0].year, scored_docs[i][0].pres, scored_docs[i][1])

scored_docs = corpus.dict_rank(our_dictionary, True, len(corpus.docs))
presidents = set([scored_doc[0].pres for scored_doc in scored_docs])
president_dictionary = {}
for president in presidents:
    scores = [scored_doc[1] for scored_doc in scored_docs if scored_doc[0].pres == president]
    president_dictionary[president] = sum(scores)/len(scores)

for pres_score in sorted(president_dictionary.items(), key=operator.itemgetter(1)):
    print "{0} {1}".format(pres_score[0].rjust(15), pres_score[1])

#################################################################################

def load_sentiment_dictionary(path):
    """
    description: load a sentiment dictionary
    input: path: the path to the dictionary
    """

    d = {}
    with open(path, 'rb') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            d[row[0]] = int(row[1])        

    return(d)

sentiment_dictionary = load_sentiment_dictionary('../data/AFINN/AFINN-111.txt')

# Inspect sentiment dictionary
print "Sentiment dictionary length: {0}".format(len(sentiment_dictionary))

def load_words_from_file(path):

    # Read the file content.
    file_handle = open(path)
    file_content = file_handle.read()
    file_handle.close()

    # Extract the content as JSON and get a copy of the speech text.
    speech = json.loads(file_content)[2]
    stripped_text = speech

    # For each nonalphanumeric character, replace with a space.  This is
    # safer than replacing with an empty string because some punctuation
    # separates words without a space, i.e. '--'.
    for char in ',.:;[]"?$:-':
        stripped_text = stripped_text.replace(char, ' ')

    # Split the string into words.
    word_list = stripped_text.split(' ')

    # Because of the way the punctuation was replaced with spaces, there are
    # instances of multiple adjacent spaces.  Therefore, empty strings appear
    # in the word list.  Remove these.
    word_list = [word.lower() for word in word_list if word != '']

    return(word_list)

def calculate_sentiment_for_speech(sentiment_dictionary, words):

    word_count = 0
    sentiment = 0

    for word in words:
        if sentiment_dictionary.has_key(word):
            sentiment += sentiment_dictionary[word]
            word_count += 1

    return float(sentiment)/float(word_count), sentiment

def print_results(sentiment_dictionary, path, friendly_string):

    words = load_words_from_file(path)
    sentiment, cumul_sent_score = calculate_sentiment_for_speech(sentiment_dictionary, words)
    display_str = "{0} : sentiment = {1} ; cumulative sentiment score = {2}"
    print display_str.format(friendly_string, sentiment, cumul_sent_score)

print_results(sentiment_dictionary,
              "../data/pres_speech/1977_Ford_Matthew.txt",
              "1977 Ford")
print_results(sentiment_dictionary,
              "../data/pres_speech/1967_Johnson_Roger.txt",
              "1967 Johnson")
print_results(sentiment_dictionary,
              "../data/pres_speech/1897_McKinley_miquel.txt",
              "1897 McKinley")

#As is evident from the data above, McKinley in 1897 had the lowest sentiment in his speech,
#while Ford in 1977 had the highest sentiment in his.
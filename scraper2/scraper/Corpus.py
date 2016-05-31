from Database import *
from Property import *
from collections import Counter
from math import log
import numpy as np
import json

class Corpus:
    
    def __init__(self):

        self.docTermMatrix = None
        self.tfIdfMatrix = None
        self.loadDocumentsAndVocabulary()

    def loadDocumentsAndVocabulary(self):

        db = Database()

        sql = u"select id, stemmed_tokens from property where usable = 1;"

        table = db.runSqlQueryTable(sql)

        self.propertyIds = [row[0] for row in table]
        self.documents = [row[1].split(" ") for row in table]

        vocab = set()

        for document in self.documents[0:50]:
            for token in document:
                vocab.add(token)
 
        self.vocabulary = sorted(vocab)

    def docTermVector(self, doc):
        """
        Description: generate a term-vector for this document.  The result
                     corresponds with a single row of the document-term-matrix
                     of the corpus
        """
        vector = [None] * len(self.vocabulary)
        counter = Counter(doc)
        for i in range(len(self.vocabulary)):
            count = counter[self.vocabulary[i]]
            vector[i] = count

        return vector

    def getDocTermMatrix(self):

        if (self.docTermMatrix != None):
            return self.docTermMatrix

        docTermMatrix = []
            
        for doc in self.documents[0:50]:
            vector = self.docTermVector(doc)
            docTermMatrix.append(vector)
            
        self.docTermMatrix = docTermMatrix  
        
        return docTermMatrix

    def getTfIdfMatrix(self):

        if (self.tfIdfMatrix != None):
            return self.tfIdfMatrix

        # Generate a copy of the document-term matrix to work with in this
        # function and initialize other local variables.
        dtMatrix = self.getDocTermMatrix()
        tfMatrix = []
        idfMatrix = []
        tfIdfMatrix = []

        # Build a term frequency matrix from the document term matrix.
        # tf(d,v) = { 0 if x(d,v) = 0, 1 + log(x(d), v) otherwise }
        for dtDocVector in dtMatrix:
            tfDocVector = [(0 if x == 0 else 1 + log(x)) for x in dtDocVector]
            tfMatrix.append(tfDocVector)

        # Build a document frequency matrix for each term.
        # Initialize with zeros.
        dfVector = np.zeros(len(self.vocabulary))
        for dtDocVector in dtMatrix:
            # Increment the counters based on an indicator function which
            # is 1 if there is at least one instance of the term in the doc.
            dfVector = np.add(dfVector, [int(x > 0) for x in dtDocVector])

        # Build an inverse document frequency vector.
        idfDocVector = [log(len(self.documents) / x) for x in dfVector]

        # Build the TF-IDF weighting matrix.
        for tfDocVector in tfMatrix:
            tfIdfVector = np.multiply(tfDocVector, idfDocVector)
            tfIdfMatrix.append(tfIdfVector)

        self.tfIdfMatrix = tfIdfMatrix

        return tfIdfMatrix

#corpus = Corpus()
#M = corpus.docTermMatrix()
#with open("C:\\OneDrive\\Career\\github\\BGSE-text-mining\\scraper2\\scraper\\docTermMatrix.data", 'wb') as f:
#    pickle.dump(M, f)
#tfidf = corpus.TF_IDF()
pass

corpus = Corpus()
M = corpus.getDocTermMatrix()
T = corpus.getTfIdfMatrix()
with open("C:\\OneDrive\\Career\\github\\BGSE-text-mining\\scraper2\\scraper\\docTermMatrix.data", 'w') as outfile:
    json.dump(T, outfile)

pass
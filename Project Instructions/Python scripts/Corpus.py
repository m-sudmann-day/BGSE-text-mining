from Database import *
from Property import *
from collections import Counter
from math import log
import numpy as np
import pickle
import lda
from scipy import sparse

class Corpus:
    
    def __init__(self):

        self.propertyIds = None
        self.documents = None
        self.vocabulary = None

    def loadDocumentsAndVocabulary(self, availabilityLag):

        print "\nLoading documents from MySQL"

        db = Database()

        sql = u"select id, test_train, stemmed_tokens from property where usable = 1 and availability_lag <= {0};"
        sql = sql.format(availabilityLag)

        table = db.runSqlQueryTable(sql)

        self.propertyIds = [row[0] for row in table]
        self.testTrain = [row[1] for row in table]
        self.documents = [row[2].split(" ") for row in table]

        print len(self.documents), " documents loaded"

        print "\nLoading vocabulary"

        vocab = set()

        for document in self.documents:
            for token in document:
                vocab.add(token)
 
        self.vocabulary = sorted(vocab)

        print len(self.vocabulary), " terms loaded"

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

        docTermMatrix = [None] * len(self.documents)

        for i in range(len(self.documents)):
            #if (i % 100 == 0):
            #    print "getDocTermMatrix", i
            doc = self.documents[i]
            docTermMatrix[i] = self.docTermVector(doc)
        
        self.docTermMatrix = docTermMatrix

        return self.docTermMatrix

    def writeDtmToFile(self, path):

        self.docTermMatrix = self.getDocTermMatrix()

        with open(path, 'wb') as outfile:
            pickle.dump(self.docTermMatrix, outfile)

    def readDtmFromFile(self, path):

        print 1

        with open(path, 'rb') as outfile:
            self.docTermMatrix = pickle.load(outfile)

        print 2

        return self.docTermMatrix

    def getTfIdfMatrix(self):

        # Generate a copy of the document-term matrix to work with in this
        # function and initialize other local variables.
        dtMatrix = self.getDocTermMatrix()
        tfMatrix = []
        idfMatrix = []
        tfIdfMatrix = []

        # Build a term frequency matrix from the document term matrix.
        # tf(d,v) = { 0 if x(d,v) = 0, 1 + log(x(d), v) otherwise }
        counter = 0            
        for dtDocVector in dtMatrix:
            counter += 1
            if (counter % 100 == 0):
                print "for dtDocVector in dtMatrix", counter
            tfDocVector = [(0 if x == 0 else 1 + log(x)) for x in dtDocVector]
            tfMatrix.append(tfDocVector)

        # Build a document frequency matrix for each term.
        # Initialize with zeros.
        counter = 0            
        dfVector = np.zeros(len(self.vocabulary))
        for dtDocVector in dtMatrix:
            counter += 1
            if (counter % 100 == 0):
                print "for dtDocVector in dtMatrix (2)", counter
            # Increment the counters based on an indicator function which
            # is 1 if there is at least one instance of the term in the doc.
            dfVector = np.add(dfVector, [int(x > 0) for x in dtDocVector])

        # Build an inverse document frequency vector.
        idfDocVector = [log(len(self.documents) / x) for x in dfVector]

        # Build the TF-IDF weighting matrix.
        counter = 0            
        for tfDocVector in tfMatrix:
            counter += 1
            if (counter % 100 == 0):
                print "for tfDocVector in tfMatrix", counter
            tfIdfVector = np.multiply(tfDocVector, idfDocVector)
            tfIdfMatrix.append(tfIdfVector)

        return np.asarray(tfIdfMatrix)

    def getTfIdfMatrix_Ints(self):

        T = self.getTfIdfMatrix()

        T = np.int_(np.multiply(T, 100))

        T = sparse.lil_matrix(T)

        return T

    def writeTfIdfMatrixToFile_Ints(self, path):

        self.tfIdfMatrix_Ints = self.getTfIdfMatrix_Ints()

        with open(path, 'wb') as outfile:
            pickle.dump(self.tfIdfMatrix_Ints, outfile)

    def readTfIdfMatrixFromFile_Ints(self, path):

        with open(path, 'rb') as infile:
            self.tfIdfMatrix_Ints = pickle.load(infile)

        return self.tfIdfMatrix_Ints

    def getSubset(self, obj, cutoff, isTest):

        if isTest:
            indices = [i for i in range(len(self.testTrain)) if self.testTrain[i] >= cutoff]
        else:
            indices = [i for i in range(len(self.testTrain)) if self.testTrain[i] < cutoff]

        if isinstance(obj, sparse.lil.lil_matrix):
            result = obj.tocsr()
            result = result[indices, :]
            return result
        else:
            return [obj[i] for i in indices]

    def assignLdaTopicProjections_TFIDF(self):

        numWordsToPrint = 20
        numIterations = 50
        numTopics = 5
        testTrainCutoff = 0.8

        self.writeTfIdfMatrixToFile_Ints("tf_idf_100.data")
        
        TFIDF = self.readTfIdfMatrixFromFile_Ints("tf_idf_100.data")

        TrainingTFIDF = self.getSubset(TFIDF, testTrainCutoff, False)
        TrainingIds = self.getSubset(self.propertyIds, testTrainCutoff, False)

        print "Fitting LDA model"

        model = lda.LDA(n_topics = numTopics, n_iter = numIterations, random_state = 1)
        model.fit(TrainingTFIDF)
        with open("model.data", 'wb') as outfile:
            pickle.dump(model, outfile)

        print "Loading LDA model"

        with open("model.data", 'rb') as infile:
            model = pickle.load(infile)

        vocab = np.array(self.vocabulary)

        for i, topic_dist in enumerate(model.topic_word_):
            print "*** Topic ", i
            print vocab[np.argsort(topic_dist)][:-numWordsToPrint:-1]

        print "Generating projections"

        projections = model.transform(TFIDF, max_iter = numIterations)
        with open("projections.data", 'wb') as outfile:
            pickle.dump(projections, outfile)

        print "Loading projections"

        with open("projections.data", 'rb') as infile:
            projections = pickle.load(infile)

        db = Database()

        print "Saving projections"

        for i in range(len(self.propertyIds)):
            projection = projections[i]
            sql = u"update property set lda1={0}, lda2={1}, lda3={2}, lda4={3}, lda5={4} where id = {5};"
            sql = sql.format(projection[0], projection[1], projection[2], projection[3], projection[4], self.propertyIds[i])
            db.runSqlCommand(sql)

        print "Done"

    def assignLdaTopicProjections(self):

        numWordsToPrint = 20
        numIterations = 50
        numTopics = 5
        testTrainCutoff = 0.8

        print "\nGenerating DTM"

        DTM = self.getDocTermMatrix()

        DTM_sparse = sparse.lil_matrix(DTM)

        print "DTM dimensions: ", DTM_sparse.shape

        TrainingDTM = self.getSubset(DTM_sparse, testTrainCutoff, False)
        TrainingIds = self.getSubset(self.propertyIds, testTrainCutoff, False)

        print "\nFitting LDA model"

        model = lda.LDA(n_topics = numTopics, n_iter = numIterations, random_state = 1)
        model.fit(TrainingDTM)

        vocab = np.array(self.vocabulary)

        print "\nLDA model fitted"
        print "\nPRESS ENTER TO SEE LDA TOPICS"
        sys.stdin.readline()

        for i, topic_dist in enumerate(model.topic_word_):
            print "\n*** Topic ", i
            print vocab[np.argsort(topic_dist)][:-numWordsToPrint:-1]

        print "\nGenerating projections"

        projections = model.transform(DTM_sparse, max_iter = numIterations)

        db = Database()

        print "\nSaving projections to MySQL"

        for i in range(len(self.propertyIds)):
            projection = projections[i]
            sql = u"update property set lda1={0}, lda2={1}, lda3={2}, lda4={3}, lda5={4} where id = {5};"
            sql = sql.format(projection[0], projection[1], projection[2], projection[3], projection[4], self.propertyIds[i])
            db.runSqlCommand(sql)

    def showLdaDetails():

        with open("model.data", 'rb') as infile:
            model = pickle.load(infile)

        vocab = np.array(corpus.vocabulary)

        for i, topic_dist in enumerate(model.topic_word_):
            print "*** Topic ", i
            print vocab[np.argsort(topic_dist)][:-20:-1]
            print topic_dist[np.argsort(topic_dist)][:-20:-1]


corpus = Corpus()
corpus.loadDocumentsAndVocabulary(21)

print "\nPRESS ENTER TO CONTINUE"
sys.stdin.readline()

corpus.assignLdaTopicProjections()

print "\nYou won't see sentiment analysis here.  It was"
print "performed when the data was scraped and saved"
print "with all the other metadata."
print "See the function analyseSentiment() in Property.py."

print "\nFINISHED - PRESS ANY KEY TO EXIT"
sys.stdin.readline()

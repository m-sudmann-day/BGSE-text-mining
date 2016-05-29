from Database import *
from Property import *

class Corpus:
    
    def __init__(self):

        self.loadVocabulary()

    def loadVocabulary(self):

        db = Database()

        sql = u"select id from property where usable = 1;"

        propertyIds = db.runSqlQueryColumn(sql)

        vocab = set()

        for propertyId in propertyIds:
            property = Property(db, propertyId)
            tokens = property.stemmedTokens.split(" ")
            for token in tokens:
                vocab.add(token)
    
        self.vocab = sorted(vocab)

corpus = Corpus()
pass

### Scraper.py
### This is the entry point script for executing the scraper.

from Database import *
from Outcode import *
from Property import *
from random import shuffle
import Utilities

try:
    from nltk.tokenize import wordpunct_tokenize
except Exception:
    pass

import pysentiment as ps
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import lda

DEFAULT_OUTCODE_IDS = "793"
#RURAL NORFOLK: = "1822"
#EDINBURGH: "793 804 815 826 837 843 844 845 846 794 795 796 797 798 799 800 801"
MAX_OUTCODE_ID = 2917

def GetNewListingsForSpecificOutcodes(daysAgo):

    # Get the outcode IDs we will scrape from the user.
    # Split and strip and make a clean list of integers.
    sys.stdout.write("Outcode IDs separated by spaces (default='" + DEFAULT_OUTCODE_IDS + "'): ")
    outcodeIds = sys.stdin.readline().strip()
    if (outcodeIds == ""):
        outcodeIds = DEFAULT_OUTCODE_IDS
    outcodeIds = outcodeIds.split(" ")
    outcodeIds = [int(outcodeId) for outcodeId in outcodeIds if outcodeId.strip() != '']

    oldestAdDate = Utilities.DaysAgo(daysAgo)

    # Create the database object which will prompt the user for connection information.
    db = Database()

    # Loop through all the outcodes and scrape each one by creating an Outcode object
    # which automatically triggers scraping from its constructor.
    for outcodeId in outcodeIds:
        Outcode(db, outcodeId, oldestAdDate)

def GetNewListingsForAllOutcodes(daysAgo, includeToday=True, ignoreIfCompleted=True):

    # Randomize the list of outcodes to make ourselves a bit less obvious.
    outcodeIds = range(MAX_OUTCODE_ID)
    shuffle(outcodeIds)

    oldestAdDate = Utilities.DaysAgo(daysAgo)

    # Create the database object which will prompt the user for connection information.
    db = Database()

    counter = 0

    # Loop through all the outcodes and scrape each one by creating an Outcode object
    # which automatically triggers scraping from its constructor.
    for outcodeId in outcodeIds:
        counter += 1
        print "Outcode counter:", counter
        Outcode(db, outcodeId, oldestAdDate, includeToday, ignoreIfCompleted)

def ScapeOneProperty(outcodeId, propertyId, isPremium):

    db = Database()

    Property(db, outcodeId, propertyId, isPremium)

def UpdateStatuses(availabilityLag):

    db = Database()

    sql = u"select id from property where usable = 1 and let_agreed = 0 and removed = 0 and availability_lag <= {0};"
    sql = sql.format(availabilityLag);

    pendingLetAgreed = db.runSqlQueryColumn(sql)

    for propertyId in pendingLetAgreed:
        property = Property(db, propertyId)
        property.checkLetAgreed()

def RunSentimentAnalysis():

    hiv4 = ps.HIV4()

    db = Database()

    sql = u"select id from property;"

    propertyIds = db.runSqlQueryColumn(sql)

    for propertyId in propertyIds:
        property = Property(db, propertyId)
        property.analyseSentiment(hiv4)

def CleanupText():

    db = Database()

    sql = u"select id from property;"

    propertyIds = db.runSqlQueryColumn(sql)

    # Portions of code copied from:
    #   Latent Dirichlet Allocation (LDA) with Python
    #   by Jordan Barber

    tokenizer = RegexpTokenizer(r'\w+')

    stopWords = get_stop_words('en')

    stemmer = PorterStemmer()

    for propertyId in propertyIds:
        property = Property(db, propertyId)
        property.cleanupText(tokenizer, stopWords, stemmer)

#GetNewListingsForAllOutcodes(3)

# The following were rerun because they threw errors the first time.
# However, in all cases except one, it was due to a lack of location
# information so they were ignored thereafter.  The one that was due
# to a network error ran fine (I had verified manually that it was
# indeed NOT premium.)
#ScapeOneProperty(2462,59617616,False)
#ScapeOneProperty(142,59609798,False)
#ScapeOneProperty(1676,24773018,False)
#ScapeOneProperty(518,42061545,False)
#ScapeOneProperty(1820,5587838,False)
#ScapeOneProperty(745,59642756,False)
#ScapeOneProperty(2463,42030936,False)
#ScapeOneProperty(485,42068847,False)
#ScapeOneProperty(508,42058230,False)
#ScapeOneProperty(1593,47573894,False)
#ScapeOneProperty(2505,5453285,False)
#ScapeOneProperty(458,42039567,False)
#ScapeOneProperty(2459,59640761,False)
#ScapeOneProperty(2459,59623847,False)
#ScapeOneProperty(1055,42068010,False)
#ScapeOneProperty(499,59662136,False)

#GetNewListingsForAllOutcodes(2, includeToday=False, ignoreIfCompleted=False)

#time.sleep(3*60*60)
#print datetime.today()
#UpdateStatuses(21)
#print datetime.today()

#RunSentimentAnalysis()

#CleanupText()

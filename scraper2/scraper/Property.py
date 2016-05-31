### Property.py
### Defines the class Property.

from bs4 import BeautifulSoup
import requests
import time
from datetime import datetime

from Database import *
from Utilities import *

class Property:

    PROPERTY_URL = "http://www.rightmove.co.uk/property-to-rent/property-{0}.html"

    def __init__(self, db, outcodeId, propertyId, isPremium):

        self.db = db
        self.propertyId = propertyId
        self.outcodeId = outcodeId
        self.isPremium = isPremium
        self.url = ""
        self.title = ""
        self.description = ""
        self.letAgreed = False
        self.monthlyPrice = None
        self.weeklyPrice = None
        self.latitude = None
        self.longitude = None
        self.lettingType = None
        self.furnishing = None
        self.dateAvailable = None
        self.dateAdvertised = None
        self.availabilityLag = None # this is set by a SQL script after scraping completes
        self.errorOccured = False

        # If the property has already been scraped, return.
        # Otherwise, scrape it and save it as a property record in the database.

        try:

            if self.isAlreadyScraped():
                print "Already scraped property", self.propertyId
                return

            self.scrape()

            self.save()
        
        except Exception as ex:

            # If an error occurs, flag the object and log it.
            
            self.errorOccured = True

            print "Error", ex

            LogError(self.db, outcodeId, propertyId, ex)

    def __init__(self, db, propertyId):

        sql = u"select * from property where id = {0}".format(propertyId)

        record = db.runSqlQueryRecord(sql)

        self.db = db
        self.propertyId = propertyId
        self.outcodeId = record["outcode_id"]
        self.isPremium = record["is_premium"]
        self.url = record["url"]
        self.title = record["title"]
        self.description = record["description"]
        self.letAgreed = record["let_agreed"]
        self.removed = record["removed"]
        self.monthlyPrice = record["monthly_price"]
        self.weeklyPrice = record["weekly_price"]
        self.latitude = record["latitude"]
        self.longitude = record["longitude"]
        self.lettingType = record["letting_type"]
        self.furnishing = record["furnishing"]
        self.dateAvailable = record["date_available"]
        self.dateAdvertised = record["date_advertised"]
        self.availabilityLag = record["availability_lag"]
        self.initialTimestamp = record["initial_timestamp"]
        self.latestTimestamp = record["latest_timestamp"]
        self.sentPol = record["sent_pol"]
        self.sentSubj = record["sent_subj"]
        self.sentPos = record["sent_pos"]
        self.sentNeg = record["sent_neg"]
        self.stemmedTokens = record["stemmed_tokens"]
        self.errorOccured = False

    def checkLetAgreed(self):

        print "Property", self.propertyId

        try:

            # Construct a URL for property.
            self.url = self.PROPERTY_URL.format(self.propertyId)

            # Retrieve the HTML content from the URL.
            content = requests.get(self.url, headers = {'User-agent': 'Mozilla/5.0'}).text

            # Let BeautifulSoup build an object model from the content.
            soup = BeautifulSoup(content, "lxml")

            invalidPropertyNodes = soup.find_all("strong", {"class":"block"})
            for node in invalidPropertyNodes:
                if "We are sorry but" in node.text:
                    self.removed = True
                    break

            if not self.removed:
                propertyHeaderQualifierNodes = soup.find_all("small", {"class":"property-header-qualifier"})
                for node in propertyHeaderQualifierNodes:
                    if node != None:
                        if "Let Agreed" in node.text:
                            self.letAgreed = True
                            break

            if self.removed or self.letAgreed:

                self.latestTimestamp = datetime.today().date()

                sql = "update property set removed = {0}, let_agreed = {1}, latest_timestamp = '{2}' where id = {3};"
                sql = sql.format(int(self.removed), int(self.letAgreed), Database.currentTimestampForDatabase(), self.propertyId)

                self.db.runSqlCommand(sql)

        except Exception as ex:

            # If an error occurs, flag the object and log it.
            
            self.errorOccured = True

            print "Error", ex

            LogError(self.db, -1, self.propertyId, ex)

    def isAlreadyScraped(self):
       
        # Execute a SQL query to identify whether or not the property has
        # already been scraped.

        sql = "select count(1) from property where id = {0};"
        sql = sql.format(self.propertyId)

        result = (self.db.runSqlQueryScalar(sql) == 1)
        return result

    def save(self):

        # Execute a SQL command to save the property to the database.

        sql = u"insert into property (id, outcode_id, url, title, description, "
        sql += "monthly_price, weekly_price, latitude, longitude, let_agreed, letting_type, furnishing, "
        sql += "date_available, date_advertised, initial_timestamp, latest_timestamp, is_premium) values "
        sql += "({0}, {1}, '{2}', '{3}', '{4}', {5}, {6}, {7}, {8}, {9}, '{10}', '{11}', {12}, {13}, '{14}', '{14}', {15});"

        sql = sql.format(self.propertyId,
                         self.outcodeId,
                         Database.cleanStringForDatabase(self.url),
                         Database.cleanStringForDatabase(self.title),
                         Database.cleanStringForDatabase(self.description),
                         Database.cleanNumberForDatabase(self.monthlyPrice),
                         Database.cleanNumberForDatabase(self.weeklyPrice),
                         Database.cleanNumberForDatabase(self.latitude),
                         Database.cleanNumberForDatabase(self.longitude),
                         Database.cleanNumberForDatabase(self.letAgreed),
                         Database.cleanStringForDatabase(self.lettingType),
                         Database.cleanStringForDatabase(self.furnishing),
                         Database.cleanDateForDatabase(self.dateAvailable),
                         Database.cleanDateForDatabase(self.dateAdvertised),
                         Database.currentTimestampForDatabase(),
                         Database.cleanNumberForDatabase(self.isPremium))

        self.db.runSqlCommand(sql)

    def scrape(self):

        print "Property", self.propertyId

        # Construct a URL for property.
        self.url = self.PROPERTY_URL.format(self.propertyId)

        # Retrieve the HTML content from the URL.
        content = requests.get(self.url, headers = {'User-agent': 'Mozilla/5.0'}).text

        # Let BeautifulSoup build an object model from the content.
        soup = BeautifulSoup(content, "lxml")

        # Extract the title of the ad.
        titleNode = soup.find("h1", {"class":"fs-22"})
        if (titleNode != None):
            self.title = titleNode.text

        # Extract the header which contains the weekly and monthly prices.
        propertyHeaderNode = soup.find("p", {"id":"propertyHeaderPrice", "class":"property-header-price"})
        propertyHeaderNodeText = propertyHeaderNode.find("strong").text

        # Use a regular expression to extract the monthly price.
        pcmMatch = MyRegex("([0-9,]+) pcm", propertyHeaderNodeText)
        if (pcmMatch != None):
            pcmMatch = pcmMatch.replace(",", "")
            self.monthlyPrice = int(pcmMatch)

        # Use a regular expression to extract the monthly price.
        pwMatch = MyRegex("([0-9,]+) pw", propertyHeaderNodeText)
        if (pwMatch != None):
            pwMatch = pwMatch.replace(",", "")
            self.weeklyPrice = int(pwMatch)

        # Extract the node that marks the beginning of the description.
        descriptionNode = soup.find("h3", text="Full description")
        descriptionNode = descriptionNode.nextSibling
        description = ""

        # Loop through all subsequent siblings of the text 'Full description'
        # to extract the actual user-created text describing the property.
        # Skip nodes that have tag names because these are not pure text.
        while descriptionNode != None:
            tagName = getattr(descriptionNode, "name", None)
            if (tagName == None):
                description += descriptionNode + " "
            descriptionNode = descriptionNode.nextSibling

        self.description = description.strip()

        # Extract the anchor for the Google map image.
        mapAnchor = soup.find("a", {"class":"block js-tab-trigger js-ga-minimap"})
        if (mapAnchor == None):
            raise Exception("Location cannot be identified.")

        mapImage = mapAnchor.find("img")["src"]

        # Use a regular expression to extract the latitude from the Google map image link.
        latitudeMatch = MyRegex("latitude=([0-9\.\-]+)", mapImage)
        if (latitudeMatch != None):
            self.latitude = float(latitudeMatch)
            
        # Use a regular expression to extract the longitude from the Google map image link.
        longitudeMatch = MyRegex("longitude=([0-9\.\-]+)", mapImage)
        if (longitudeMatch != None):
            self.longitude = float(longitudeMatch)

        strongNodes = soup.find_all("strong")
        for strongNode in strongNodes:
            s = strongNode.text.lower().strip()
            if (s == "letting type:"):
                valueNode = strongNode.parent.nextSibling.nextSibling
                self.lettingType = valueNode.text
            elif (s == "furnishing:"):
                valueNode = strongNode.parent.nextSibling.nextSibling
                self.furnishing = valueNode.text
            elif (s == "date available:"):
                valueNode = strongNode.parent.nextSibling.nextSibling
                self.dateAvailable = MyDateParser(valueNode.text)
            elif (s == "added on rightmove:"):
                valueNode = strongNode.parent.nextSibling.nextSibling
                self.dateAdvertised = MyDateParser(valueNode.text)

        propertyHeaderQualifierNodes = soup.find_all("small", {"class":"property-header-qualifier"})
        for node in propertyHeaderQualifierNodes:
            if node != None:
                if "Let Agreed" in node.text:
                    self.letAgreed = True

    def analyseSentiment(self, hiv4):

        tokens = hiv4.tokenize(self.description)

        score = hiv4.get_score(tokens)

        sql = "update property set sent_pol = {0}, sent_subj = {1}, sent_pos = {2}, sent_neg = {3} where id = {4};"
        sql = sql.format(score['Polarity'], score['Subjectivity'], score['Positive'], score['Negative'], self.propertyId)

        self.db.runSqlCommand(sql)

    def cleanupText(self, tokenizer, stopWords, stemmer):

        text = self.description
        text = text.lower()

        tokens = tokenizer.tokenize(text)
        tokens = [ConvertToAscii(s).strip().strip('-').strip('_') for s in tokens]
        tokens = [s for s in tokens if not s in stopWords]
        tokens = [s for s in tokens if not any(c.isdigit() for c in s)]

        self.num_tokens = len(tokens)

        stemmedTokens = [stemmer.stem(s).strip() for s in tokens]
        stemmedTokens = [s for s in stemmedTokens if len(s) > 0]

        if len(stemmedTokens) < 12:
            stemmedTokens = None
            sql = "update property set stemmed_tokens = NULL, num_tokens = {0} where id = {1};"
            sql = sql.format(self.num_tokens, self.propertyId)
        else:
            stemmedTokens = ' '.join(stemmedTokens).strip()
            sql = "update property set stemmed_tokens = '{0}', num_tokens = {1} where id = {2};"
            sql = sql.format(stemmedTokens, self.num_tokens, self.propertyId)

        self.stemmedTokens = stemmedTokens

        self.db.runSqlCommand(sql)

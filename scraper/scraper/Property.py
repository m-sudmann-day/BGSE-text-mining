### Property.py
### Defines the class Property.

from bs4 import BeautifulSoup
import requests
import time

from Database import *
from Utilities import *

class Property:

    PROPERTY_URL = "http://www.rightmove.co.uk/property-to-rent/property-{0}.html"

    def __init__(self, db, propertyId, outcodeId):

        self.db = db
        self.propertyId = propertyId
        self.outcodeId = outcodeId
        self.url = ""
        self.title = ""
        self.description = ""
        self.monthlyPrice = None
        self.weeklyPrice = None
        self.latitude = None
        self.longitude = None
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

            LogError(self.db, "Property", propertyId, ex)

    def isAlreadyScraped(self):
       
        # Execute a SQL query to identify whether or not the property has
        # already been scraped.

        sql = "select count(1) from property where id = {0};"
        sql = sql.format(self.propertyId)

        result = (self.db.runSqlQuery(sql) == 1)
        return result

    def save(self):

        # Execute a SQL command to save the property to the database.

        sql = u"insert into property (id, outcode_id, url, title, description, "
        sql += "monthly_price, weekly_price, latitude, longitude) values "
        sql += "({0}, {1}, '{2}', '{3}', '{4}', {5}, {6}, {7}, {8});"

        sql = sql.format(self.propertyId,
                         self.outcodeId,
                         Database.cleanStringForDatabase(self.url),
                         Database.cleanStringForDatabase(self.title),
                         Database.cleanStringForDatabase(self.description),
                         Database.cleanNumberForDatabase(self.monthlyPrice),
                         Database.cleanNumberForDatabase(self.weeklyPrice),
                         Database.cleanNumberForDatabase(self.latitude),
                         Database.cleanNumberForDatabase(self.longitude))

        self.db.runSqlCommand(sql)

    def scrape(self):

        print "Property", self.propertyId

        # Construct a URL for property.
        self.url = self.PROPERTY_URL.format(self.propertyId)

        # Retrieve the HTML content from the URL.
        content = requests.get(self.url).text

        # Let BeautifulSoup build an object model from the content.
        soup = BeautifulSoup(content, "lxml")

        # Extract the title of the ad.
        self.title = soup.find("h1").text

        # Extract the header which contains the weekly and monthly prices.
        propertyHeaderNode = soup.find("p", {"id":"propertyHeaderPrice", "class":"property-header-price"})
        propertyHeaderTextNode = propertyHeaderNode.find("strong").text

        # Use a regular expression to extract the monthly price.
        pcmMatch = MyRegex("([0-9,]+) pcm", propertyHeaderTextNode)
        if (pcmMatch != None):
            pcmMatch = pcmMatch.replace(",", "")
            self.monthlyPrice = int(pcmMatch)

        # Use a regular expression to extract the monthly price.
        pwMatch = MyRegex("([0-9,]+) pw", propertyHeaderTextNode)
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
        mapImage = mapAnchor.find("img")["src"]

        # Use a regular expression to extract the latitude from the Google map image link.
        latitudeMatch = MyRegex("latitude=([0-9\.\-]+)", mapImage)
        if (latitudeMatch != None):
            self.latitude = float(latitudeMatch)
            
        # Use a regular expression to extract the longitude from the Google map image link.
        longitudeMatch = MyRegex("longitude=([0-9\.\-]+)", mapImage)
        if (longitudeMatch != None):
            self.longitude = float(longitudeMatch)

    def prettyPrint(self):

        # For diagnostic purposes, provide a function for pretty printing this object.

        print "Id:", self.propertyId
        print "Price per month:", self.monthlyPrice
        print "Price per week:", self.weeklyPrice
        print "Latitude:", self.latitude
        print "Longitude:", self.longitude
        print "Title:", self.title
        print "Description:", self.description

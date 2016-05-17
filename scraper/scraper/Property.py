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

        if self.isAlreadyScraped():
            print "Already scraped property", self.propertyId
            return

        self.scrape()

        self.save()

    def isAlreadyScraped(self):
       
        sql = "select count(1) from property where id = {0};"
        sql = sql.format(self.propertyId)

        result = (self.db.runSqlQuery(sql) == 1)
        return result

    def save(self):

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

        self.url = self.PROPERTY_URL.format(self.propertyId)

        content = requests.get(self.url).text

        soup = BeautifulSoup(content, "lxml")

        self.title = Database.cleanStringForDatabase(soup.find("h1").text)

        propertyHeaderNode = soup.find("p", {"id":"propertyHeaderPrice", "class":"property-header-price"})
        propertyHeaderTextNode = propertyHeaderNode.find("strong").text

        pcmMatch = MyRegex("([0-9]+) pcm", propertyHeaderTextNode)
        if (pcmMatch != None):
            self.monthlyPrice = int(pcmMatch)

        pwMatch = MyRegex("([0-9]+) pw", propertyHeaderTextNode)
        if (pwMatch != None):
            self.weeklyPrice = int(pwMatch)

        descriptionNode = soup.find("h3", text="Full description")
        descriptionNode = descriptionNode.nextSibling
        description = ""

        while descriptionNode != None:
            tagName = getattr(descriptionNode, "name", None)
            if (tagName == None):
                description += descriptionNode
            descriptionNode = descriptionNode.nextSibling

        self.description = Database.cleanStringForDatabase(description.strip())

        mapAnchor = soup.find("a", {"class":"block js-tab-trigger js-ga-minimap"})
        mapImage = mapAnchor.find("img")["src"]

        latitudeMatch = MyRegex("latitude=([0-9\.\-]+)", mapImage)
        if (latitudeMatch != None):
            self.latitude = float(latitudeMatch)
            
        longitudeMatch = MyRegex("longitude=([0-9\.\-]+)", mapImage)
        if (longitudeMatch != None):
            self.longitude = float(longitudeMatch)

    def prettyPrint(self):
        print "Id:", self.propertyId
        print "Price per month:", self.monthlyPrice
        print "Price per week:", self.weeklyPrice
        print "Latitude:", self.latitude
        print "Longitude:", self.longitude
        print "Title:", self.title
        print "Description:", self.description

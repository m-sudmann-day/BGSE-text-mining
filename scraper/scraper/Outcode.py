from bs4 import BeautifulSoup
import requests

from Database import *
from Utilities import *
from Property import *

class Outcode:
    
    SEARCH_RESULTS_URL = "http://www.rightmove.co.uk/property-to-rent/find.html?locationIdentifier=OUTCODE%5E{0}&numberOfPropertiesPerPage=50&index={1}"
    RESULTS_PER_PAGE = 50

    def __init__(self, db, outcodeId):

        self.db = db
        self.outcodeId = outcodeId
        self.properties = []

        if self.isFinished():
            return

        try:

            self.start()

            self.scrape()

            self.finish()
        
        except Exception as ex:
            
            LogError(self.db, "Outcode", outcodeId, ex)

    def start(self):

        sql = "insert into outcode (id, completed) select {0}, 0 from dual "
        sql += "where not exists (select 1 from outcode where id = {0});"
        sql = sql.format(self.outcodeId)

        self.db.runSqlCommand(sql)

    def isFinished(self):

        sql = "select completed from outcode where id = {0};"
        sql = sql.format(self.outcodeId)

        result = bool(self.db.runSqlQuery(sql))
        return result

    def finish(self):

        sql = "update outcode set completed = 1 where id = {0};"
        sql = sql.format(self.outcodeId)

        self.db.runSqlCommand(sql)

    def scrape(self):

        index = 0
        accumulatedPropertyIds = []

        while (True):
        
            print "====== Scraping outcode", self.outcodeId
            print "=== Search results page", (index / self.RESULTS_PER_PAGE) + 1

            url = self.SEARCH_RESULTS_URL.format(self.outcodeId, index)
            
            content = requests.get(url).text

            soup = BeautifulSoup(content, "lxml")

            links = soup.find_all("a")
            propertyIds = []

            for link in links:
                if link.has_attr("href"):
                    propertyId = MyRegex("/property\-to\-rent/property\-([0-9]+)\.html", link["href"])
                    if propertyId != None and len(propertyId) > 0 and (propertyId not in propertyIds):
                        propertyIds.append(propertyId)

            propertyIds = [propertyId for propertyId in propertyIds if propertyId not in accumulatedPropertyIds]
            
            accumulatedPropertyIds.extend(propertyIds)

            if len(propertyIds) == 0:
                break

            for propertyId in propertyIds:

                property = Property(self.db, propertyId, self.outcodeId)
                self.properties.append(property)

                time.sleep(2)

            index = index + self.RESULTS_PER_PAGE

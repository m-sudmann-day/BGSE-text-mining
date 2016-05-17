### Outcode.py
### Defines the class Outcode.

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
        self.errorOccured = False

        print "====== Scraping outcode", self.outcodeId

        if self.isFinished():
            print "Previously successfully scraped"
            return
        
        try:

            # Create a database record for the outcode.
            self.start()

            # Scrape.
            self.scrape()

            # If no properties had errors, flag the database record as completed.
            if not self.errorOccured:
                self.finish()
        
        except Exception as ex:
            
            # If an error is thrown, log it.  The outcode will not be marked as completed.
            LogError(self.db, "Outcode", outcodeId, ex)

    def start(self):

        # Insert an outcode record.

        sql = "insert into outcode (id, completed) select {0}, 0 from dual "
        sql += "where not exists (select 1 from outcode where id = {0});"
        sql = sql.format(self.outcodeId)

        self.db.runSqlCommand(sql)

    def isFinished(self):

        # Check to see if an existing outcode record, if any, is finished.

        sql = "select completed from outcode where id = {0};"
        sql = sql.format(self.outcodeId)

        result = bool(self.db.runSqlQuery(sql))
        return result

    def finish(self):

        # Mark the outcode record as finished.

        sql = "update outcode set completed = 1 where id = {0};"
        sql = sql.format(self.outcodeId)

        self.db.runSqlCommand(sql)

    def scrape(self):

        # Initialize the index of the first property on the first search results page.
        index = 0

        # Keep track of all property IDs scraped for this outcode in all iterations,
        # i.e. in each page of search results.
        accumulatedPropertyIds = []

        # Loop through all pages of search results.
        while (True):
        
            print "=== Search results page", (index / self.RESULTS_PER_PAGE) + 1

            # Construct a URL for this page of search results.
            url = self.SEARCH_RESULTS_URL.format(self.outcodeId, index)
            
            # Retrieve the HTML content from the URL.
            content = requests.get(url).text

            # Let BeautifulSoup build an object model from the content.
            soup = BeautifulSoup(content, "lxml")

            # Get a list of all links in the document.  We should be able to
            # limit the results using other attributes, but this did not seem stable.
            links = soup.find_all("a")
            propertyIds = []

            # For each link that has a URL matching the property detail page, extract
            # the property ID.
            for link in links:
                if link.has_attr("href"):
                    propertyId = MyRegex("/property\-to\-rent/property\-([0-9]+)\.html", link["href"])
                    if propertyId != None and len(propertyId) > 0 and (propertyId not in propertyIds):
                        propertyIds.append(propertyId)

            # Eliminate property IDs that we have already seen.  (Some featured properties
            # appear on every page.)
            propertyIds = [propertyId for propertyId in propertyIds if propertyId not in accumulatedPropertyIds]
            
            # Extend our list of all property IDs for this outcode by the new ones found on this page.
            accumulatedPropertyIds.extend(propertyIds)

            # If no new property IDs are found, exit the loop.  This is what happens
            # AFTER the last page of properties.  Our loop always goes one "page" beyond
            # what a human user of the website would see.
            if len(propertyIds) == 0:
                print "No new properties on this page"
                break

            # For each property ID, create a property.
            for propertyId in propertyIds:

                property = Property(self.db, propertyId, self.outcodeId)

                # If there's an error, flag it.  Otherwise append this property to the
                # list of properties of this outcode, just in case we ever actually need
                # an object model.
                if property.errorOccured:
                    self.errorOccured = True
                else:
                    self.properties.append(property)

                # Sleep for an extra two seconds to take it easy on the website.
                time.sleep(2)

            # Increment the index of the first result on a page.
            index = index + self.RESULTS_PER_PAGE

### Outcode.py
### Defines the class Outcode.

from bs4 import BeautifulSoup
import requests
from datetime import datetime
from datetime import timedelta

from Database import *
from Utilities import *
from Property import *

class Outcode:
    
    SEARCH_RESULTS_URL = "http://www.rightmove.co.uk/property-to-rent/find.html?locationIdentifier=OUTCODE%5E{0}&includeLetAgreed=true&sortType=6&numberOfPropertiesPerPage=50&viewType=LIST&index={1}"
    RESULTS_PER_PAGE = 50

    def __init__(self, db, outcodeId, oldestAdDate, includeToday, ignoreIfCompleted):

        self.db = db
        self.outcodeId = outcodeId
        self.oldestAdDate = oldestAdDate
        self.includeToday = includeToday
        self.properties = []
        self.errorOccured = False

        print "====== Scraping outcode", self.outcodeId

        if ignoreIfCompleted and self.isFinished():
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
            LogError(self.db, outcodeId, NULL, ex)

    def start(self):

        # Insert an outcode record.

        sql = "insert into outcode (id, completed, timestamp) select {0}, 0, '{1}' from dual "
        sql += "where not exists (select 1 from outcode where id = {0});"
        sql = sql.format(self.outcodeId, Database.currentTimestampForDatabase())

        self.db.runSqlCommand(sql)

    def isFinished(self):

        # Check to see if an existing outcode record, if any, is finished.

        sql = "select completed from outcode where id = {0};"
        sql = sql.format(self.outcodeId)

        result = bool(self.db.runSqlQueryScalar(sql))
        return result

    def finish(self):

        # Mark the outcode record as finished.

        sql = "update outcode set completed = 1, timestamp = '{1}' where id = {0};"
        sql = sql.format(self.outcodeId, Database.currentTimestampForDatabase())

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
            content = requests.get(url, headers = {'User-agent': 'Mozilla/5.0'}).text

            # Let BeautifulSoup build an object model from the content.
            soup = BeautifulSoup(content, "lxml")

            finishedWithSearchResults = False
            propertyIds = []
            propertyCards = soup.find_all("div",{"class":"propertyCard propertyCard--premium  "})
            propertyCards.extend(soup.find_all("div",{"class":"propertyCard  "}))

            print "Num property cards:", len(propertyCards)

            for propertyCard in propertyCards:

                isPremium = ("propertyCard--premium" in propertyCard.attrs['class'])

                # Look at the branch summary information which says when the ad was added (unless
                # it was revised, in which case it will be ignored).
                branchSummary = propertyCard.find("div", {"class":"propertyCard-branchSummary"})
                addDate = None
                if branchSummary != None:
                    branchSummaryText = branchSummary.text.strip().lower()
                    if (branchSummaryText.startswith("added")):
                        if (branchSummaryText.startswith("added today")):
                            if not self.includeToday:
                                break
                            addDate = MyDateParser("today")
                        elif (branchSummaryText.startswith("added yesterday")):
                            addDate = MyDateParser("yesterday")
                        else:
                            addDate = MyDateParser(branchSummaryText[9:])

                # Because the search results sort by date descending, if we find an "added" date
                # that is earlier than our cutoff, quit the loop.  Unless it's a premium property
                # which could be at the top of the page.
                if addDate != None and addDate < self.oldestAdDate and not isPremium:
                    finishedWithSearchResults = True
                    break

                if addDate != None and addDate >= self.oldestAdDate:
                    # For each link that has a URL matching the property detail page, extract
                    # the property ID.
                    links = propertyCard.find_all("a", {"class":"propertyCard-link"})
                    for link in links:
                        if link.has_attr("href"):
                            # This regex explicitly excludes commercial properties because their URLs are different.
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

                # Sleep for an extra two seconds per property.
                time.sleep(1)

                property = Property(self.db, self.outcodeId, propertyId, isPremium)

                # If there's an error, flag it.  Otherwise append this property to the
                # list of properties of this outcode, just in case we ever actually need
                # an object model.
                if property.errorOccured:
                    self.errorOccured = True
                else:
                    self.properties.append(property)

            # Sleep for an extra two seconds after scraping a page of search results.
            time.sleep(1)

            # There's nothing left to search for in this outcode.
            if finishedWithSearchResults:
                break

            # Increment the index of the first result on a page.
            index = index + self.RESULTS_PER_PAGE



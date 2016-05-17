### Scraper.py
### This is the entry point script for executing the scraper.

from Database import *
from Outcode import *

DEFAULT_OUTCODE_IDS = "1822"
#RURAL NORFOLK: = "1822"
#EDINBURGH: "793 804 815 826 837 843 844 845 846 794 795 796 797 798 799 800 801"

# Get the outcode IDs we will scrape from the user.
# Split and strip and make a clean list of integers.
sys.stdout.write("Outcode IDs separated by spaces (default='" + DEFAULT_OUTCODE_IDS + "'): ")
outcodeIds = sys.stdin.readline().strip()
if (outcodeIds == ""):
    outcodeIds = DEFAULT_OUTCODE_IDS
outcodeIds = outcodeIds.split(" ")
outcodeIds = [int(outcodeId) for outcodeId in outcodeIds if outcodeId.strip() != '']

# Create the database object which will prompt the user for connection information.
db = Database()

# Loop through all the outcodes and scrape each one by creating an Outcode object
# which automatically triggers scraping from its constructor.
for outcodeId in outcodeIds:
    Outcode(db, outcodeId)

print "====== Scraping completed"

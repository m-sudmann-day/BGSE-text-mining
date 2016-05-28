### Utilities.py
### Defines utilty functions for general use.

import re

from Database import *
from datetime import datetime
from datetime import timedelta

def MyRegex(pattern, text):

    # A function for extracting regular expression matches using some parameters
    # that are appropriate for all uses by this application.

    matches = re.findall(pattern, text, re.MULTILINE|re.DOTALL)
    
    if len(matches) == 1:
        return matches[0]
    else:
        return None

def LogError(db, outcodeId, propertyId, ex):
   
    # Log an error to the database.

    print "Error", outcodeId, propertyId
    print ex

    try:
        
        sql = u"insert into error (outcode_id, property_id, text, timestamp) "
        sql += u"values ({0}, {1}, '{2}', '{3}')"

        sql = sql.format(outcodeId,
                         propertyId,
                         Database.cleanStringForDatabase(str(ex)),
                         Database.currentTimestampForDatabase())

        db.runSqlCommand(sql)

    except:
        pass

def MyDateParser(s):

    s = s.strip().lower()
    
    if (s == "today" or s == "now"):
        return datetime.today().date()
    if (s == "yesterday"):
        return (datetime.today() - timedelta(days=1)).date()
    if ('/' in s):
        return datetime.strptime(s[0:10], u"%d/%m/%Y").date()

    pos = s.index(" (")
    s = s[0:pos]
    return datetime.strptime(s, u"%d %B %Y").date()

def DaysAgo(n):

    return (datetime.today() - timedelta(days=n)).date()

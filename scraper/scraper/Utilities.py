### Utilities.py
### Defines utilty functions for general use.

import re

from Database import *

def MyRegex(pattern, text):

    # A function for extracting regular expression matches using some parameters
    # that are appropriate for all uses by this application.

    matches = re.findall(pattern, text, re.MULTILINE|re.DOTALL)
    
    if len(matches) == 1:
        return matches[0]
    else:
        return None

def LogError(db, objType, id, ex):
   
    # Log an error to the database.

    print "Error", objType, id
    print ex

    try:
        
        sql = u"insert into error (obj_type, id, text) values ('{0}', '{1}', '{2}')"

        sql = sql.format(objType, id, Database.cleanStringForDatabase(str(ex)))

        db.runSqlCommand(sql)

    except:
        pass
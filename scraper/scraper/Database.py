### Database.py
### Defines the class Database.

import sys
import MySQLdb

class Database:

    # Provide a constructor that allows the caller to fully define the
    # MySQL connection.
    def __init__(self, host, user, passwd, db):
   
        self.host = host
        self.user = user
        self.passwd = passwd
        self.db = db

    # Provide a constructor that will prompt the user of a console application
    # for the attributes of the MySQL connection with defaults.
    def __init__(self):
        sys.stdout.write("MySQL host (default='127.0.0.1'): ")
        self.host = sys.stdin.readline().strip()
        if (self.host == ""):
            self.host = "127.0.0.1"

        sys.stdout.write("MySQL user (default='root'): ")
        self.user = sys.stdin.readline().strip()
        if (self.user == ""):
            self.user = "root"

        sys.stdout.write("MySQL password (default='root'): ")
        self.passwd = sys.stdin.readline().strip()
        if (self.passwd == ""):
            self.passwd = "root"

        sys.stdout.write("MySQL database (default='scraper'): ")
        self.db = sys.stdin.readline().strip()
        if (self.db == ""):
            self.db = "scraper"

    def getConnection(self):

        # Create and return a MySQLdb connection.

        conn = MySQLdb.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.db)
        return conn

    def runSqlQuery(self, sql):

        # Execute a SQL query and return a scalar result.

        sql = sql.encode("utf8")
        conn = self.getConnection()
        query = conn.query(sql)
        result = conn.use_result()
        row = result.fetch_row()
        if (len(row) > 0 and len(row[0]) > 0):
            value = row[0][0]
        else:
            value = None
        conn.close()
        return value

    def runSqlCommand(self, sql):

        # Execute a SQL query with no return value.

        sql = sql.encode("utf8")
        conn = self.getConnection()
        query = conn.query(sql)
        conn.commit()
        conn.close()

    # A static helper method for cleaning a string before it is inserted into
    # a dynamic SQL string.
    @staticmethod
    def cleanStringForDatabase(s):
        if (s == None):
            return ""
        else:
            return s.replace("'", "").replace("\"", "")

    # A static helper method for cleaning a supposedly numeric value before
    # it is inserted into a dynamic SQL string.
    @staticmethod
    def cleanNumberForDatabase(n):
        if (n == None):
            return "NULL"
        else:
            return n
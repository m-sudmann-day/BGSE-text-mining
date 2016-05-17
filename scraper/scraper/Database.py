import sys
import MySQLdb

class Database:

    def __init__(self, host, user, passwd, db):
   
        self.host = host
        self.user = user
        self.passwd = passwd
        self.db = db

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

        conn = MySQLdb.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.db)
        return conn

    def runSqlQuery(self, sql):

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

        sql = sql.encode("utf8")
        conn = self.getConnection()
        query = conn.query(sql)
        conn.commit()
        conn.close()

    @staticmethod
    def cleanStringForDatabase(s):
        if (s == None):
            return ""
        else:
            return s.replace("'", "").replace("\"", "")

    @staticmethod
    def cleanNumberForDatabase(n):
        if (n == None):
            return "NULL"
        else:
            return n
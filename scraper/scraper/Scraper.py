from Database import *
from Outcode import *

DEFAULT_OUTCODE_IDS = "793 804 815 826 837 843 844 845 846 794 795 796 797 798 799 800 801"

sys.stdout.write("MySQL host (default='" + DEFAULT_OUTCODE_IDS + "'): ")
outcodeIds = sys.stdin.readline().strip()
if (outcodeIds == ""):
    outcodeIds = DEFAULT_OUTCODE_IDS
outcodeIds = outcodeIds.split(" ")
outcodeIds = [int(outcodeId) for outcodeId in outcodeIds if outcodeId.strip() != '']


db = Database()

for outcodeId in outcodeIds:
    Outcode(db, outcodeId)

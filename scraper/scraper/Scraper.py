from Database import *
from Utilities import *
from Property import *
from Outcode import *

OUTCODE_IDS = [793, 804, 815, 826, 837, 843, 844, 845, 846, 794, 795, 796, 797, 798, 799, 800, 801]

db = Database()

for outcodeId in OUTCODE_IDS:
    Outcode(db, outcodeId)

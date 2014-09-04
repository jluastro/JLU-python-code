from pysqlite2 import dbapi2 as sqlite
from gcwork import starTables
import numpy as np

def loadOldStars():
    """
    Loads list of known old stars from database, which is
    continuously updated (per T. Do).
    """
    dbfile = '/u/ghezgroup/data/gc/database/stars.sqlite'

    # Create a connection to the database file
    connection = sqlite.connect(dbfile)

    # Create a cursor object
    cur = connection.cursor()

    cur.execute('SELECT * FROM stars')
    oldNames = []
    for row in cur:
        old = row[3]
        if (old == 'T'):
            oldNames = np.concatenate([oldNames,[row[0]]])
        
    return oldNames

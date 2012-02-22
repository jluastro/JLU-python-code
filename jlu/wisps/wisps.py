import asciidata

def read_coords():
    """
    Read in the list of coordinates for the WISPS fields.
    """
    cooFile = '/u/jlu/work/wisps/WISPS_hmsdms.coords'

    cooTable = asciidata.open(cooFile, delimiter=',')

    ra = cooTable.tonumpy()
    dec = cooTable.tonumpy()

    # Need to query SIMBAD

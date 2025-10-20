import os
from pykoa.koa import Koa
from astropy.table import Table

def download_for_mulab(instrument, date, cookie_file):
    """
    Download OSIRIS data for a single night and put it into the
    expected location for MULab data.

    Please note that you need to login before running this routine.
    Here is how you login:

    from pykoa.koa import Koa
    cookie_file = '/u/jlu/keck_tap_cookie.txt')
    Koa.login(cookie_file)

    And enter your username and password.

    """
    # Login
    Koa.login(cookie_file)
    
    # Setup useful variables.
    inst_low = instrument.lower()
    inst_upp = instrument.upper()

    date_hyphens = date
    date_nohyphs = date.replace("_", "")
    
    data_dir = f'/g/lu/data/KECK/{date_nohyphs}_{inst_upp}/'
    data_table = data_dir + 'data.tbl'

    # Make the data directory.
    os.makedirs(data_dir, exist_ok=True)

    # Query for the list of data available for download.
    # Must be logged in to get proprietary data for the night. 
    Koa.query_date(inst_low, date_hyphens, data_table, format='ipac',
                   cookiepath=cookie_file)

    # Fetch the individual data files. 
    Koa.download(data_table, 'ipac', data_dir, calibdir=0, calibfile=1,
                 cookiepath=cookie_file)
    
    tab = Table.read(data_table, format='ipac')

    for ii in range(len(tab)):
        root_old = f'{data_dir}/lev0/{tab[ii]["koaid"].replace(".fits", "")}' 
        root_new = f'{data_dir}/lev0/{tab[ii]["ofname"].replace(".fits", "")}' 
        
        os.rename(root_old + '.fits',
                  root_new + '.fits')

        if os.path.exists(root_old + '.caliblist.tbl'):
            os.rename(root_old + '.caliblist.tbl',
                      root_new + '.caliblist.tbl')
        if os.path.exists(root_old + 'caliblist.json'):
            os.rename(root_old + '.caliblist.json',
                      root_new + '.caliblist.json')
    
    return

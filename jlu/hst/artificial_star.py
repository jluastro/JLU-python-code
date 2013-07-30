import numpy as np
import pylab as py
import atpy
from jlu.hst import starlists

def combine_in_out_ks2(ks2_input_file, matchup_files, suffixes=None):
    """
    Read in a LOGA.INPUT file and the corresponding MATCHUP.XYMEEE files
    and cross-match them. They should already be the same length and the in same
    order.

    Inputs
    ------
    ks2_input_file -- the LOGA.INPUT file
    matchup_files -- a list of MATCHUP.XYMEEE files

    Optional Inputs
    ---------------
    suffixes -- a list (same length as matchup_files) 
    """

    final = atpy.Table()
    final.table_name = ''

    # Read in the input ks2 file
    inp = atpy.Table('LOGA.INPUT', type='ascii')

    final.add_column('name', inp.col17)
    final.add_column('x_in', inp.col1)
    final.add_column('y_in', inp.col2)

    for ff in range(len(matchup_files)):
        if suffixes == None:
            suf = '_{0}'.format(ff+1)
        else:
            suf = '_' + suffixes[ff]

        f_in = 'col{0}'.format(18 + 4*ff)
        m_in = -2.5 * np.log10(inp[f_in])
        
        final.add_column('m_in' + suf, m_in)

        match = starlists.read_matchup(matchup_files[ff])

        if len(match) != len(final):
            print 'Problem reading ' + matchup_files[ff] + ': incorrect file length'

        final.add_column('x_out' + suf, match.x)
        final.add_column('y_out' + suf, match.y)
        final.add_column('m_out' + suf, match.m)
        final.add_column('xe_out' + suf, match.xe)
        final.add_column('ye_out' + suf, match.ye)
        final.add_column('me_out' + suf, match.me)


    final.write('ks2_in_out_catalog.fits', overwrite=True)

    
    


    

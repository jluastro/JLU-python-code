import numpy as np
from astropy.io.votable import parse

def process_arches_votable():
    """
    Executed this query:
    SELECT TOP 500 * FROM gaiadr1.gaia_source  WHERE CONTAINS(POINT('ICRS',gaiadr1.gaia_source.ra,gaiadr1.gaia_source.dec),CIRCLE('ICRS',266.46042,-28.82444,0.016666666666666666))=1  

    on the gaia archive:
    http://gea.esac.esa.int/archive/

    Saved to result.vot (in Downloads)
    """

    result_file = '/Users/jlu/Downloads/result.vot'

    tab = parse(result_file)

    foo = tab.get_first_table()

    # this is an astropy table
    data_tab = foo.to_table()

    data_tab.write('/Users/jlu/work/tmt/iris/arches_gaia.txt', format='ascii.fixed_width', delimiter=' ')

    return

    
def two_point_solve(x, y):
    """
    Calculate the scale and angle from two measured points.
    dE = scale * (dX cos a - dY sin a)
    dN = scale * (dX sin a + dY cos a)

    scale = sqrt( (dE**2 + dN**2) / (dX**2 + dY**2) )
    angle = arctan( (dN * dX - dE * dY) / (dE * dX + dN * dY) )
    """
    e_pos = np.array([0.0, 0.0])
    n_pos = np.array([10.0, -10.0])
    x_pos = np.array([614.0, 952.0])
    y_pos = np.array([4723.0, 5377.])

    de = np.diff(e_pos)
    dn = np.diff(n_pos)
    dx = np.diff(x_pos)
    dy = np.diff(y_pos)

    scale = ((de**2 + dn**2) / (dx**2 + dy**2))**0.5

    angle = np.arctan( (dn*dx - de*dy) / (de*dx + dn*dy) )

    print( scale,  np.degrees(angle))
    


    
def xy_to_en(x, y):
    """
    Calculate the East North offsets for a desired X Y offset.
    """
    return


def process_gc_votable():
    """
    Executed this query:
    SELECT TOP 500 * FROM gaiadr1.gaia_source  WHERE CONTAINS(POINT('ICRS',gaiadr1.gaia_source.ra,gaiadr1.gaia_source.dec),CIRCLE('ICRS',266.41683333333333,-29.00777777777778,0.016666666666666666))=1  

    on the gaia archive:
    http://gea.esac.esa.int/archive/

    Saved to gc_result.vot (in Downloads)
    """

    result_file = '/Users/jlu/work/tmt/iris/gc_result.vot'

    tab = parse(result_file)

    foo = tab.get_first_table()

    # this is an astropy table
    data_tab = foo.to_table()

    data_tab.write('/Users/jlu/work/tmt/iris/gc_gaia.txt', format='ascii.fixed_width', delimiter=' ')

    return


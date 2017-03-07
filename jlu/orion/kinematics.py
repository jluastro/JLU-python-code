from astropy.table import Table
from jlu.util import constants as cc
import pylab as py
import numpy as np
from astropy.table import Table, join

def plot_dynamics():
    root_dir = '/Users/jlu/work/orion/kinematics/'
    table_file = root_dir + 'star_velocities_2016_03_18.txt'

    tab = Table.read(table_file, format='ascii')

    # Convert velocities to common units
    distance = 392.0 # pc

    tab['v_ra'] = masyr_to_kms(tab['PM_ra'], distance)
    tab['v_dec'] = masyr_to_kms(tab['PM_de'], distance)
    tab['ve_ra'] = masyr_to_kms(tab['PMe_ra'], distance)
    tab['ve_dec'] = masyr_to_kms(tab['PMe_de'], distance)

    # Extract positions into usable units.
    ra_decimal = tab['rh'] + (tab['rm'] / 60.0) + (tab['rs'] / 3600.0) * 360.0 / 24.0
    de_decimal = tab['dd'] + (tab['dm'] / 60.0) + (tab['ds'] / 3600.0)

    ra_mean = ra_decimal.mean()
    de_mean = de_decimal.mean()

    ra_asec = (ra_decimal - ra_mean) * 60.0
    de_asec = (de_decimal - de_mean) * 60.0

    # Calculate dispersions.
    v_pm_bias = (tab['ve_ra']**2 + tab['ve_dec']**2).sum() / (2.0 * (len(tab) - 1))
    v_z_bias = (tab['RVe']**2).sum() / (len(tab) - 1)

    v_pm_disp_meas = np.hypot(tab['v_ra'].std(), tab['v_dec'].std())
    v_z_disp_meas = tab['RV'].std()

    v_pm_disp = (v_pm_disp_meas**2 - v_pm_bias)**0.5
    v_z_disp = (v_z_disp_meas**2 - v_z_bias)**0.5

    print( 'Number of Stars: ', len(tab))
    print( '')
    fmt = '{0:10s}  {1:10.3f}   {2:10.3f}'
    print( '{0:10s}  {1:10s}   {2:10s}'.format('', 'v_PM (km/s)', 'v_z (km/s)'))
    print( '{0:10s}  {1:10s}   {2:10s}'.format('', '-----------', '----------'))
    print( fmt.format('Measured: ', v_pm_disp_meas, v_z_disp_meas))
    print( fmt.format('Bias: ',     v_pm_bias**0.5, v_z_bias**0.5))
    print( fmt.format('Intrinsic:', v_pm_disp, v_z_disp))
    

    bins = np.arange(-20, 40, 2.5)
    py.figure(1)
    py.clf()
    py.hist(tab['v_ra'], bins=bins, histtype='step', label='v_RA')
    py.hist(tab['v_dec'], bins=bins, histtype='step', label='v_Dec')
    py.hist(tab['RV'], bins=bins, histtype='step', label='v_z')
    py.xlabel('Velocity (km/s)')
    py.ylabel('Number of Stars')
    py.legend()
    py.savefig(root_dir + 'vel_hist.png')

    vtot = (tab['v_ra']**2 + tab['v_dec']**2 + tab['RV']**2)**0.5
    py.clf()
    py.hist(vtot, bins=bins, histtype='step')
    py.xlabel('Total Velocity (km/s)')
    py.ylabel('Number of Stars')
    py.legend()
    py.savefig(root_dir + 'vel_tot_hist.png')
    
    py.figure(2)
    py.clf()
    py.quiver(ra_asec, de_asec, tab['v_ra'], tab['v_dec'], tab['RV'], scale=100)
    py.colorbar(label='v_z (km/s)')
    py.quiver([1.2, 1.2], [1.2, 1.2], [1.0, 1.0], [0, 0], scale=100, color='black')
    py.axis('equal')
    ax = py.axis()
    py.xlim(ax[1], ax[0])
    py.xlabel('R.A. Offset (arcmin)')
    py.ylabel('Dec. Offset (arcmin)')
    py.savefig(root_dir + 'vel_vectors.png')
    
    return tab

def masyr_to_kms(prop_mot, distance):
    """
    Proper motion in milli-arcseoncds/year.
    Distance in pc.

    Output:
    Velocity in km/s
    """
    
    vel = prop_mot * distance * cc.cm_in_AU / (1.0e5 * 1.0e3 * cc.sec_in_yr)

    return vel


def combine_dynamics_tables():
    root_dir = '/Users/jlu/work/orion/kinematics/'
    table_file1 = root_dir + 'star_velocities_2016_03_18.txt'
    table_file2 = root_dir + 'PM_table.csv'
    
    t1 = Table.read(table_file1, format='ascii')
    t2 = Table.read(table_file2, format='ascii')

    # Clean up the tables 
    t2.rename_column(t2.colnames[0], 'ID')
    t1.remove_columns(['PM_ra', 'PMe_ra', 'PM_de', 'PMe_de',
                           'rh', 'rm', 'rs',
                           'dd', 'dm', 'ds'])

    # Divide the proper motions into separate columns
    pm_r = np.zeros(len(t2), dtype=float)
    pm_d = np.zeros(len(t2), dtype=float)
    pme_r = np.zeros(len(t2), dtype=float)
    pme_d = np.zeros(len(t2), dtype=float)
    for ii in range(len(t2)):
        tmp_r = t2[t2.colnames[3]][ii].split()
        pm_r[ii] = tmp_r[0]
        pme_r[ii] = tmp_r[-1]

        tmp_d = t2[t2.colnames[4]][ii].split()
        pm_d[ii] = tmp_d[0]
        pme_d[ii] = tmp_d[-1]

    t2['pm_r'] = pm_r
    t2['pm_d'] = pm_d
    t2['pme_r'] = pme_r
    t2['pme_d'] = pme_d

    t2.remove_columns([t2.colnames[3], t2.colnames[4]])
        
    tab = join(t2, t1, keys='ID', join_type='left')

    tab.write(root_dir + 'current_dynamics.fits')

    return


def plot_current_dynamics():
    root_dir = '/Users/jlu/work/orion/kinematics/'
    table_file = root_dir + 'current_dynamics.fits'

    tab = Table.read(table_file)

    # Convert velocities to common units
    distance = 392.0 # pc

    tab['v_ra'] = masyr_to_kms(tab['pm_r'], distance)
    tab['v_dec'] = masyr_to_kms(tab['pm_d'], distance)
    tab['ve_ra'] = masyr_to_kms(tab['pme_r'], distance)
    tab['ve_dec'] = masyr_to_kms(tab['pme_d'], distance)

    print('1 mas/yr = {0:.1f} km/s'.format(masyr_to_kms(1.0, distance)))

    # Extract positions into usable units.
    ra_decimal = tab['RA']
    de_decimal = tab['Dec']

    ra_mean = ra_decimal.mean()
    de_mean = de_decimal.mean()

    ra_amin = (ra_decimal - ra_mean) * 60.0
    de_amin = (de_decimal - de_mean) * 60.0

    no_rv = np.where((np.isnan(tab['RV']) == True) & (tab['pme_r'] < 1) & (tab['pme_d'] < 1))[0]
    with_rv = np.where(np.isnan(tab['RV']) == False)[0]
    
    py.figure(1, figsize=(7,6))
    py.clf()
    py.subplots_adjust(left=0.15, top=0.95, right=0.98)
    # Note, we have to flip the RA velocities cuz. invert axis won't do it.
    q = py.quiver(ra_amin[no_rv], de_amin[no_rv], -tab['v_ra'][no_rv], tab['v_dec'][no_rv],
                  color='black', scale=100, alpha=0.3)
    py.quiver(ra_amin[with_rv], de_amin[with_rv], -tab['v_ra'][with_rv], tab['v_dec'][with_rv],
                  tab['RV'][with_rv], scale=100)
    py.colorbar(label='RV (km/s)')
    py.quiver([1.2, 1.2], [1.2, 1.2], [8.0, 8.0], [0, 0], scale=100, color='black')
    py.text(1.2, 1.25, '8 km/s')
    # py.quiverkey(q, 1.2, 1.2, 8.0, label='8 km/s')
    py.axis('equal')
    py.axis('equal')
    ax = py.gca().invert_xaxis()
    py.xlabel('R.A. Offset (arcmin)')
    py.ylabel('Dec. Offset (arcmin)')
    py.savefig(root_dir + 'current_dynamics_vectors.png')
    
    return tab

    


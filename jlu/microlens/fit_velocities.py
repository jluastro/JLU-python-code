import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.table import Table, Column
from jlu.papers import lu_2019_lens as lu
import copy
import time
import pdb

astrom_data = lu.astrom_data

res_dict = {'ob120169': 1.1, 'ob140613': 0.4, 'ob150029': 1.1, 'ob150211': 1.3}
time_cuts = {'ob120169': 2012, 'ob140613': 2015, 'ob150029': 2015, 'ob150211': 2015}

class StarTable(Table):
    """
    An astropy Table initialized from a file.
    Main purpose is to fit velocities to the table's starlist.

    Input:
    target: type str - Target name (lowercase).
    """
    def __init__(self, target):
        self.target = target
        tab = Table.read(astrom_data[self.target])
        Table.__init__(self, tab)

        self['name'] = self['name'].astype('U20')

        return

    def fit(self, bootstrap=0, time_cut=None, verbose=False):
        """
        Fit velocities for all stars in the self.
        Inputting a time_cut will ignore the data of that year in the fit.
        """
        N_stars, N_epochs = self['x'].shape

        if verbose:
            start_time = time.time()
            msg = 'Starting startable.fit_velocities for {0:d} stars with n={1:d} bootstrap'
            print(msg.format(N_stars, bootstrap))

        # Clean/remove up old arrays.
        if 'x0' in self.colnames: self.remove_column('x0')
        if 'vx' in self.colnames: self.remove_column('vx')
        if 'y0' in self.colnames: self.remove_column('y0')
        if 'vy' in self.colnames: self.remove_column('vy')
        if 'x0e' in self.colnames: self.remove_column('x0e')
        if 'vxe' in self.colnames: self.remove_column('vxe')
        if 'y0e' in self.colnames: self.remove_column('y0e')
        if 'vye' in self.colnames: self.remove_column('vye')
        if 't0' in self.colnames: self.remove_column('t0')
        if 'n_vfit' in self.colnames: self.remove_column('n_vfit')

        # Define output arrays for the best-fit parameters.
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'x0'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'vx'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'y0'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'vy'))

        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'x0e'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'vxe'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'y0e'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'vye'))

        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 't0'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=int), name = 'n_vfit'))

        # Catch the case when there is only a single epoch. Just return 0 velocity
        # and the same input position for the x0/y0.
        if self['x'].shape[1] == 1:
            self['x0'] = self['x'][:,0]
            self['y0'] = self['y'][:,0]

            if 't' in self.colnames:
                self['t0'] = self['t'][:, 0]
            else:
                self['t0'] = self.meta['list_times'][0]

            if 'xe' in self.colnames:
                self['x0e'] = self['xe'][:,0]
                self['y0e'] = self['ye'][:,0]

            self['n_vfit'] = 1

            return

        # STARS LOOP through the stars and work on them 1 at a time.
        # This is slow; but robust.
        for ss in range(N_stars):
            self.fit_velocity_for_star(ss, bootstrap=bootstrap, time_cut=time_cut)

        if verbose:
            stop_time = time.time()
            print('startable.fit_velocities runtime = {0:.0f} s for {1:d} stars'.format(stop_time - start_time, N_stars))

        return

    def fit_velocity_for_star(self, ss, bootstrap=False, time_cut=None):
        def poly_model(time, *params):
            pos = np.polynomial.polynomial.polyval(time, params)
            return pos

        x = self['x'][ss, :].data
        y = self['y'][ss, :].data

        if 'xe' in self.colnames:
            xe = self['xe'][ss, :].data
            ye = self['ye'][ss, :].data
        else:
            xe = np.ones(N_epochs, dtype=float)
            ye = np.ones(N_epochs, dtype=float)

        if 't' in self.colnames:
            t = self['t'][ss, :].data
        else:
            t = self.meta['list_times']

        # Figure out where we have detections (as indicated by error columns)
        if time_cut is None:
            good = np.where((xe != 0) & (ye != 0) &
                            np.isfinite(xe) & np.isfinite(ye) &
                            np.isfinite(x) & np.isfinite(y))[0]
        else:
            good = np.where((xe != 0) & (ye != 0) &
                            np.isfinite(xe) & np.isfinite(ye) &
                            np.isfinite(x) & np.isfinite(y) &
                            (np.floor(t) != time_cut))[0]

        N_good = len(good)

        # Catch the case where there is NO good data.
        if N_good == 0:
            return

        # Everything below has N_good >= 1
        x = x[good]
        y = y[good]
        t = t[good]
        xe = xe[good]
        ye = ye[good]

        # np.polynomial ordering
        p0x = np.array([x.mean(), 0.0])
        p0y = np.array([y.mean(), 0.0])

        # Calculate the t0 for all the stars.
        t_weight = 1.0 / np.hypot(xe, ye)
        t0 = np.average(t, weights=t_weight)
        dt = t - t0

        self['t0'][ss] = t0
        self['n_vfit'][ss] = N_good

        # Catch the case where all the times are identical
        if (dt == dt[0]).all():
            wgt_x = (1.0/xe)**2
            wgt_y = (1.0/ye)**2

            self['x0'][ss] = np.average(x, weights=wgt_x)
            self['y0'][ss] = np.average(y, weights=wgt_y)
            self['x0e'][ss] = np.sqrt(np.average((x - self['x0'][ss])**2, weights=wgt_x))
            self['y0e'][ss] = np.sqrt(np.average((y - self['y0'][ss])**2, weights=wgt_x))

            self['vx'][ss] = 0.0
            self['vy'][ss] = 0.0
            self['vxe'][ss] = 0.0
            self['vye'][ss] = 0.0

            return


        # Catch the case where we have enough measurements to actually
        # fit a velocity!
        if N_good > 2:
            vx_opt, vx_cov = curve_fit(poly_model, dt, x, p0=p0x, sigma=xe)
            vy_opt, vy_cov = curve_fit(poly_model, dt, y, p0=p0y, sigma=ye)

            self['x0'][ss] = vx_opt[0]
            self['vx'][ss] = vx_opt[1]
            self['y0'][ss] = vy_opt[0]
            self['vy'][ss] = vy_opt[1]

            # Run the bootstrap
            if bootstrap > 0:
                edx = np.arange(N_good, dtype=int)

                fit_x0_b = np.zeros(bootstrap, dtype=float)
                fit_vx_b = np.zeros(bootstrap, dtype=float)
                fit_y0_b = np.zeros(bootstrap, dtype=float)
                fit_vy_b = np.zeros(bootstrap, dtype=float)

                for bb in range(bootstrap):
                    bdx = np.random.choice(edx, N_good)

                    vx_opt_b, vx_cov_b = curve_fit(poly_model, dt[bdx], x[bdx], p0=vx_opt, sigma=xe[bdx])
                    vy_opt_b, vy_cov_b = curve_fit(poly_model, dt[bdx], y[bdx], p0=vy_opt, sigma=ye[bdx])

                    fit_x0_b[bb] = vx_opt_b[0]
                    fit_vx_b[bb] = vx_opt_b[1]
                    fit_y0_b[bb] = vy_opt_b[0]
                    fit_vy_b[bb] = vy_opt_b[1]

                # Save the errors from the bootstrap
                self['x0e'][ss] = fit_x0_b.std()
                self['vxe'][ss] = fit_vx_b.std()
                self['y0e'][ss] = fit_y0_b.std()
                self['vye'][ss] = fit_vy_b.std()
            else:
                vx_err = np.sqrt(vx_cov.diagonal())
                vy_err = np.sqrt(vy_cov.diagonal())

                self['x0e'][ss] = vx_err[0]
                self['vxe'][ss] = vx_err[1]
                self['y0e'][ss] = vy_err[0]
                self['vye'][ss] = vy_err[1]

        elif N_good == 2:
            # Note nough epochs to fit a velocity.
            self['x0'][ss] = np.average(x, weights=1.0/xe**2)
            self['y0'][ss] = np.average(y, weights=1.0/ye)

            dx = np.diff(x)[0]
            dy = np.diff(y)[0]
            dt_diff = np.diff(dt)[0]

            self['x0e'][ss] = np.abs(dx) / 2**0.5
            self['y0e'][ss] = np.abs(dy) / 2**0.5
            self['vx'][ss] = dx / dt_diff
            self['vy'][ss] = dy / dt_diff
            self['vxe'][ss] = 0.0
            self['vye'][ss] = 0.0

        else:
            # N_good == 1 case
            self['n_vfit'][ss] = 1
            self['x0'][ss] = x[0]
            self['y0'][ss] = y[0]

            if 'xe' in self.colnames:
                self['x0e'] = xe[0]
                self['y0e'] = ye[0]

        return

    def plot_fit(self, fign=0, return_res=False):
        res_rng = res_dict[self.target]

        stars = np.append([self.target], lu.comp_stars[self.target])

        # Figure out the min/max of the times for these sources.
        tdx = np.where(self['name'] == self.target)[0][0]
        tmin = self['t'][tdx].min() - 0.5   # in days
        tmax = self['t'][tdx].max() + 0.5   # in days

        # Setup figure and color scales
        figsize = (13, 7.5)
        if fign != 0:
            fig = plt.figure(fign, figsize=figsize)
        else:
            fig = plt.figure(figsize=figsize)
        plt.clf()
        grid_t = plt.GridSpec(1, 3, hspace=5.0, wspace=0.5, bottom=0.60, top=0.95, left=0.12, right=0.86)
        grid_b = plt.GridSpec(2, 3, hspace=0.1, wspace=0.5, bottom=0.10, top=0.45, left=0.12, right=0.86)

        cmap = plt.cm.plasma
        norm = plt.Normalize(vmin=tmin, vmax=tmax)
        smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        smap.set_array([])

        def plot_each_star(star_num, star_name):
            # Make two boxes for each star
            ax_sky = fig.add_subplot(grid_t[0, star_num])
            ax_resX = fig.add_subplot(grid_b[1, star_num])
            ax_resY = fig.add_subplot(grid_b[0, star_num])

            # Fetch the data
            tdx = np.where(self['name'] == star_name)[0][0]
            star = self[tdx]

            # Change signs of the East
            x = star['x']*-1.0
            x0 = star['x0']*-1.0
            vx = star['vx']*-1.0

            # Make the model curves
            tmod = np.arange(tmin, tmax, 0.1)
            xmod = x0 + vx * (tmod - star['t0'])
            ymod = star['y0'] + star['vy'] * (tmod - star['t0'])
            xmode = np.hypot(star['x0e'], star['vxe'] * (tmod - star['t0']))
            ymode = np.hypot(star['y0e'], star['vye'] * (tmod - star['t0']))

            xmod_at_t = x0 + vx * (star['t'] - star['t0'])
            ymod_at_t = star['y0'] + star['vy'] * (star['t'] - star['t0'])

            # Plot Positions on Sky
            ax_sky.plot(xmod, ymod, 'k-', color='grey', zorder=1)
            ax_sky.plot(xmod + xmode, ymod + ymode, 'k--', color='grey', zorder=1)
            ax_sky.plot(xmod - xmode, ymod - ymode, 'k--', color='grey', zorder=1)
            sc = ax_sky.scatter(x, star['y'], c=star['t'], cmap=cmap, norm=norm, s=20, zorder=2)
            ax_sky.errorbar(x, star['y'], xerr=star['xe'], yerr=star['ye'],
                                ecolor=smap.to_rgba(star['t']), fmt='none', elinewidth=2, zorder=2)
            ax_sky.set_aspect('equal', adjustable='datalim')

            # Figure out which axis has the bigger data range.
            xrng = np.abs(x.max() - x.min())
            yrng = np.abs(star['y'].max() - star['y'].min())
            if xrng > yrng:
                ax_sky.set_xlim(x.min() - 0.001, x.max() + 0.001)
            else:
                ax_sky.set_ylim(star['y'].min() - 0.001, star['y'].max() + 0.001)

            # Set labels
            ax_sky.invert_xaxis()
            ax_sky.set_title(star_name.upper())
            ax_sky.set_xlabel(r'$\Delta\alpha*$ (")')
            if star_num == 0:
                ax_sky.set_ylabel(r'$\Delta\delta$ (")')

            # Plot Residuals vs. Time
            xres = (x - xmod_at_t) * 1e3
            yres = (star['y'] - ymod_at_t) * 1e3
            xrese = star['xe'] * 1e3
            yrese = star['ye'] * 1e3
            ax_resX.errorbar(star['t'], xres, yerr=xrese, fmt='r.', label=r'$\alpha*$', elinewidth=2)
            ax_resY.errorbar(star['t'], yres, yerr=yrese, fmt='b.', label=r'$\delta$', elinewidth=2)
            ax_resX.plot(tmod, xmod - xmod, 'r-')
            ax_resX.plot(tmod, xmode*1e3, 'r--')
            ax_resX.plot(tmod, -xmode*1e3, 'r--')
            ax_resY.plot(tmod, ymod - ymod, 'b-')
            ax_resY.plot(tmod, ymode*1e3, 'b--')
            ax_resY.plot(tmod, -ymode*1e3, 'b--')
            ax_resX.set_xlabel('Date (yr)')
            ax_resX.set_ylim(-res_rng, res_rng)
            ax_resY.set_ylim(-res_rng, res_rng)
            ax_resY.get_xaxis().set_visible(False)
            if star_num == 0:
                ax_resX.set_ylabel(r'$\alpha^*$')
                ax_resY.set_ylabel(r'$\delta$')
                plt.gcf().text(0.015, 0.3, 'Residuals (mas)', rotation=90, fontsize=24,
                                   ha='center', va='center')

            xmode_at_t = np.hypot(star['x0e'], star['vxe'] * (star['t'] - star['t0']))*1e3
            ymode_at_t = np.hypot(star['y0e'], star['vye'] * (star['t'] - star['t0']))*1e3

            return np.vstack([[xres], [np.hypot(xrese, xmode_at_t)]]), np.vstack([[yres],\
                             [np.hypot(yrese, ymode_at_t)]]), sc

        xr, yr, sc = plot_each_star(0, stars[0])
        sc = plot_each_star(1, stars[1])[-1]
        sc = plot_each_star(2, stars[2])[-1]
        cb_ax = fig.add_axes([0.88, 0.60, 0.02, 0.35])
        plt.colorbar(sc, cax=cb_ax, label='Year')

        plt.show()

        if return_res:
            return xr, yr

    def compare_linear_motion(self, return_results=False, fign_start=1):
        """
        Compare the linear motion of the target by first fitting linear motion to all
        the astrometry, then fitting to only non-peak astrometry.
        Calculates the significance of the signal as
             sigma^2_{average deviation} = 1 / (sum_i 1/sigma^2_{deviation, i}) / N_obs,
        where
             sigma^2_{deviation, i} = deltaX_i
        """
        # Plot the linear fit from the astrometry
        self.plot_fit(fign=fign_start)
        # Plot the linear fit without the peak year and get the residuals
        time_cut = time_cuts[self.target]
        self.fit(time_cut=time_cut)
        xr, yr = self.plot_fit(fign=(fign_start + 1), return_res=True)

        tdx = np.where(self['name'] == self.target)[0][0]
        idx = np.where(np.floor(self[tdx]['t']) == time_cut)[0]

        xres = xr[0][idx]
        xrese = xr[1][idx]
        yres = yr[0][idx]
        yrese = yr[1][idx]
        rres = np.hypot(xres, yres)
        rrese = np.hypot(xres/rres * xrese, yres/rres * yrese)

        weights = 1/rrese
        average = np.multiply(rres, weights).sum() / weights.sum()
        var = 1/np.power(weights, 2).sum()/len(idx)

        print("N = %d"%len(idx))
        print("average deviation = %.2f +/- %.2f mas"%(average, np.sqrt(var)))

        if return_results:
            return average, var

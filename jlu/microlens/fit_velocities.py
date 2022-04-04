import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.table import Table, Column
from jlu.papers import lu_2019_lens as lu
import copy
import time
import pdb

astrom_data = lu.astrom_data

time_cuts = {'ob120169': 2012, 'ob140613': 2015, 'ob150029': 2015, 'ob150211': 2015}
res_dict = {'ob120169': 1.1, 'ob140613': 0.6, 'ob150029': 1.1, 'ob150211': 1.3}

class StarTable(Table):
    """
    An astropy Table initialized from a file.
    Main purpose is to fit velocities to the table's starlist.

    Input:
    target: type str - Target name (lowercase).
    """
    def __init__(self, target=None, masked=False):
        if target == None:
            Table.__init__(self)
            
            # self['name'] = self['name'].astype('U20')
            # self.cut = self.cut
            # self.time_cut = self.time_cut
        else:
            self.target = target
            tab = Table.read(astrom_data[self.target])
            Table.__init__(self, tab)

            self['name'] = self['name'].astype('U20')
            self.cut = False # Tracks whether the data has been cut or not
            self.time_cut = time_cuts[target]

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

        if time_cut is not None:
            self.cut = True

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

    def calc_chi2(self, star):
        ss = np.where(self['name']==star)[0][0]

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

        # Find the data used in the fit
        if self.cut:
            time_cut = time_cuts[self.target]
            good = np.where((xe != 0) & (ye != 0) &
                            np.isfinite(xe) & np.isfinite(ye) &
                            np.isfinite(x) & np.isfinite(y) &
                            (np.floor(t) != time_cut))[0]
        else:
            good = np.where((xe != 0) & (ye != 0) &
                            np.isfinite(xe) & np.isfinite(ye) &
                            np.isfinite(x) & np.isfinite(y))[0]

        x = x[good]
        y = y[good]
        t = t[good]
        xe = xe[good]
        ye = ye[good]

        # Change signs of the East                                                                        
        x = x*-1.0
        x0 = self['x0'][ss]*-1.0
        y0 = self['y0'][ss]
        vx = self['vx'][ss]*-1.0
        vy = self['vy'][ss]
        
        xmod_at_t = x0 + vx * (t - self['t0'][ss])
        ymod_at_t = y0 + vy * (t - self['t0'][ss])
        xres = (x - xmod_at_t)
        yres = (y - ymod_at_t)

        # Calculate chi^2
        chi2_x = xres**2 / xe**2
        chi2_y = yres**2 / ye**2

        chi2 = np.sum(chi2_x + chi2_y)

        N_data = len(x) + len(y)
        N_param = 4
        N_dof = N_data - N_param
        
        chi2_red = chi2 / N_dof
        print(star, self.cut)

        return chi2, chi2_red

    def plot_fit(self, title, fign=1, return_res=False, save_plot=False):
        '''
        Plot the linear fit of the target and companion stars.
        
        Parameters
        ----------
        title : string
            String (usually "all" or "off_peak") for the figure
            to be added to the title "_astrometry".
        fign : int, optional
            Figure number.
            Default is 1.
        return_res : bool, optional
            If True, returns the residuals to the fit for the target.
            Default is False.
        save_plot : bool, optional
            If True, saves the figure.
            Default is False.
            
        Returns
        -------
        xr, yr : array_like, optional
            If return_res is True, the residuals in x and y
            are returned, each array containing the residuals
            and their error.
        '''
        res_rng = res_dict[self.target]

        stars = np.append([self.target], lu.comp_stars[self.target])

        # Figure out the min/max of the times for these sources.
        tdx = np.where(self['name'] == self.target)[0][0]
        tmin = self['t'][tdx].min() - 0.5   # in days
        tmax = self['t'][tdx].max() + 0.5   # in days

        # Setup figure and color scales
        figsize = (13, 9.5)
        if fign != 0:
            fig = plt.figure(fign, figsize=figsize)
        else:
            fig = plt.figure(figsize=figsize)
        plt.clf()

        st = fig.suptitle(title + " astrometry", fontsize = 20)

        grid_t = plt.GridSpec(1, 3, hspace=5.0, wspace=0.5, bottom=0.60, top=0.90, left=0.12, right=0.86)
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
            y = star['y']
            x0 = star['x0']*-1.0
            y0 = star['y0']
            vx = star['vx']*-1.0
            vy = star['vy']

            # Make the model curves
            tmod = np.arange(tmin, tmax, 0.1)
            xmod = x0 + vx * (tmod - star['t0'])
            ymod = y0 + vy * (tmod - star['t0'])
            xmode = np.hypot(star['x0e'], star['vxe'] * (tmod - star['t0']))
            ymode = np.hypot(star['y0e'], star['vye'] * (tmod - star['t0']))

            xmod_at_t = x0 + vx * (star['t'] - star['t0'])
            ymod_at_t = y0 + vy * (star['t'] - star['t0'])

            # Plot Positions on Sky
            ax_sky.plot(xmod, ymod, '-', color='grey', zorder=1)
            ax_sky.plot(xmod + xmode, ymod + ymode, '--', color='grey', zorder=1)
            ax_sky.plot(xmod - xmode, ymod - ymode, '--', color='grey', zorder=1)
            sc = ax_sky.scatter(x, y, c=star['t'], cmap=cmap, norm=norm, s=20, zorder=2)
            ax_sky.errorbar(x, y, xerr=star['xe'], yerr=star['ye'],
                                ecolor=smap.to_rgba(star['t']), fmt='none', elinewidth=2, zorder=2)
            ax_sky.set_aspect('equal', adjustable='datalim')

            # Figure out which axis has the bigger data range.
            xrng = np.abs(x.max() - x.min())
            yrng = np.abs(y.max() - y.min())
            if xrng > yrng:
                ax_sky.set_xlim(x.min() - 0.001, x.max() + 0.001)
            else:
                ax_sky.set_ylim(y.min() - 0.001, y.max() + 0.001)

            # Set labels
            ax_sky.invert_xaxis()
            ax_sky.set_title(star_name.upper())
            ax_sky.set_xlabel(r'$\Delta\alpha*$ (")')
            if star_num == 0:
                ax_sky.set_ylabel(r'$\Delta\delta$ (")')

            # Plot Residuals vs. Time
            xres = (x - xmod_at_t) * 1e3
            yres = (y - ymod_at_t) * 1e3
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
                plt.gcf().text(0.015, 0.3, 'Residuals (mas)', rotation=90, fontsize=18,
                                   ha='center', va='center')

            xmode_at_t = np.hypot(star['x0e'], star['vxe'] * (star['t'] - star['t0']))*1e3
            ymode_at_t = np.hypot(star['y0e'], star['vye'] * (star['t'] - star['t0']))*1e3
            x_err = np.hypot(xrese, xmode_at_t)
            y_err = np.hypot(yrese, ymode_at_t)

            return np.vstack([[xres], [x_err]]), np.vstack([[yres], [y_err]]), sc
            
        xr, yr, sc = plot_each_star(0, stars[0])
        sc = plot_each_star(1, stars[1])[-1]
        sc = plot_each_star(2, stars[2])[-1]
        cb_ax = fig.add_axes([0.88, 0.60, 0.02, 0.35])
        plt.colorbar(sc, cax=cb_ax, label='Year')

        if save_plot:
            plt.savefig("{0}_{1}_linear_fit.pdf".format(self.target, title))

        if return_res:
            return xr, yr

    def plot_target(self, fign=1, save=True):
        '''
        Plots the linear fit for the target only. The figure is saved.
        
        Parameters
        ----------
        fign : int, optional
            Figure number.
            Default is 1.
        save : bool, optional
            Save the figure.
            Default is True.
        '''
        res_rng = res_dict[self.target]

        stars = np.append([self.target], lu.comp_stars[self.target])

        # Figure out the min/max of the times for these sources.
        tdx = np.where(self['name'] == self.target)[0][0]
        tmin = self['t'][tdx].min() - 0.5   # in days
        tmax = self['t'][tdx].max() + 0.5   # in days

        # Setup figure and color scales
        figsize = (6, 9.5)
        if fign != 0:
            fig = plt.figure(fign, figsize=figsize)
        else:
            fig = plt.figure(figsize=figsize)
        plt.clf()

        # st = fig.suptitle(self.target + " astrometry", fontsize = 20)

        grid_t = plt.GridSpec(1, 1, hspace=3.0, wspace=0.5, bottom=0.60, top=0.90, left=0.3, right=0.79)
        grid_b = plt.GridSpec(2, 1, hspace=0.1, wspace=0.5, bottom=0.10, top=0.45, left=0.3, right=0.79)

        cmap = plt.cm.plasma
        norm = plt.Normalize(vmin=tmin, vmax=tmax)
        smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        smap.set_array([])

        ax_sky = fig.add_subplot(grid_t[0, 0])
        ax_resX = fig.add_subplot(grid_b[1, 0])
        ax_resY = fig.add_subplot(grid_b[0, 0])

        # Fetch the data
        tdx = np.where(self['name'] == self.target)[0][0]
        star = self[tdx]

        # Change signs of the East
        x = star['x']*-1.0
        y = star['y']
        x0 = star['x0']*-1.0
        y0 = star['y0']
        vx = star['vx']*-1.0
        vy = star['vy']
        
        # Make the model curves
        tmod = np.arange(tmin, tmax, 0.1)
        xmod = x0 + vx * (tmod - star['t0'])
        ymod = y0 + vy * (tmod - star['t0'])
        xmode = np.hypot(star['x0e'], star['vxe'] * (tmod - star['t0']))
        ymode = np.hypot(star['y0e'], star['vye'] * (tmod - star['t0']))

        xmod_at_t = x0 + vx * (star['t'] - star['t0'])
        ymod_at_t = y0 + vy * (star['t'] - star['t0'])

        # Plot Positions on Sky
        ax_sky.plot(xmod, ymod, '-', color='grey', zorder=1)
        ax_sky.plot(xmod + xmode, ymod + ymode, '--', color='grey', zorder=1)
        ax_sky.plot(xmod - xmode, ymod - ymode, '--', color='grey', zorder=1)
        sc = ax_sky.scatter(x, y, c=star['t'], cmap=cmap, norm=norm, s=20, zorder=2)
        ax_sky.errorbar(x, y, xerr=star['xe'], yerr=star['ye'],
                            ecolor=smap.to_rgba(star['t']), fmt='none', elinewidth=2, zorder=2)
        ax_sky.set_aspect('equal', adjustable='datalim')

        # Figure out which axis has the bigger data range.
        xrng = np.abs(x.max() - x.min())
        yrng = np.abs(y.max() - y.min())
        if xrng > yrng:
            ax_sky.set_xlim(x.min() - 0.001, x.max() + 0.001)
        else:
            ax_sky.set_ylim(y.min() - 0.001, y.max() + 0.001)

        # Set labels
        ax_sky.invert_xaxis()
        ax_sky.set_title(self.target.upper())
        ax_sky.set_xlabel(r'$\Delta\alpha*$ (")')
        ax_sky.set_ylabel(r'$\Delta\delta$ (")')

        # Plot Residuals vs. Time
        xres = (x - xmod_at_t) * 1e3
        yres = (y - ymod_at_t) * 1e3
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

        xresrng = xres + np.sign(xres)*(xrese + 0.1)
        ax_resX.set_ylim(xresrng.min(), xresrng.max())
        yresrng = yres + np.sign(yres)*(yrese + 0.1)
        ax_resY.set_ylim(yresrng.min(), yresrng.max())
        ax_resY.get_xaxis().set_visible(False)
        ax_resX.set_ylabel(r'$\alpha^*$')
        ax_resY.set_ylabel(r'$\delta$')
        plt.gcf().text(0.04, 0.3, 'Residuals (mas)', rotation=90, fontsize=18,
                           ha='center', va='center')

        cb_ax = fig.add_axes([0.8, 0.60, 0.02, 0.3])
        plt.colorbar(sc, cax=cb_ax, label='Year')

        if save:
            plt.savefig("{}_linear_fit.pdf".format(self.target))

    def compare_linear_motion(self, return_results=False, fign_start=1, save_all=False):
        """
        Compare the linear motion of the target by first fitting linear motion to all
        the astrometry, then fitting to only non-peak astrometry.
        Calculates the significance of the signal as the ratio of the the average
        deviation during the peak year to the root of it error,
             sigma_{average deviation} = sqrt{ 1 / (sum_i 1/sigma^2_{deviation, i}) / N_obs },
        where
             sigma^2_{deviation, i} = deltaR_i.
        These numbers are printed, along with the number N_obs of observations in the peak year.

        Parameters
        ----------
        self : astropy.table
        return_results : bool, optional
            If True, returns the average deviation and its variance.
            Default is False.
        fign_start : int, optional
            The first figure number (for the linear fit of all the astrometry).
            Used to not overplot on different targets.
            Default is 1.
        save_all : bool, optional
            If True, saves all plots (comparison plots of both linear fits
            the target-only off-peak fit).
            If False, only the target-only off-peak fit is saved.
            Default is False.

        Returns
        -------
        average : float, optional
            Returned if return_results = True.
            The average deviation of the astrometry in the peak year in mas.
        var : float, optional
            Returned if return_results = True.
            The variance of the average above in mas^2.
        """
        # Plot the linear fit from the astrometry analysis
        self.plot_fit(title = 'all', fign=fign_start, save_plot=save_all)
        all_chi2, all_chi2_red = self.calc_chi2(self.target)
        
        # Plot the linear fit without the peak year and get the residuals
        time_cut = time_cuts[self.target]
        self.fit(time_cut=self.time_cut) # Fit without the peak year
        # Plot the fit for the target. This figure will always be saved.
        self.plot_target(fign=(fign_start+1))
        xr, yr = self.plot_fit(title = 'off_peak', fign=(fign_start + 2),
                               return_res=True, save_plot=save_all)
        cut_chi2, cut_chi2_red = self.calc_chi2(self.target)

        tdx = np.where(self['name'] == self.target)[0][0]
        idx = np.where(np.floor(self[tdx]['t']) == time_cut)[0]

        xres = xr[0][idx]
        xrese = xr[1][idx]
        yres = yr[0][idx]
        yrese = yr[1][idx]
        rres = np.hypot(xres, yres)
        rrese = np.hypot(xres/rres * xrese, yres/rres * yrese)

        weights = 1 / rrese**2
        average = np.average(rres, weights=weights)
        variance = np.average((rres-average)**2, weights=weights)

        if return_results:
            return average, variance, np.array([all_chi2, all_chi2_red]), np.array([cut_chi2, cut_chi2_red])
        else:
            print("N_obs = %d"%len(idx))
            print("average deviation = %.2f +/- %.2f mas"%(average, np.sqrt(variance)))
            print("     chi^2       chi^2_red")
            print("all: %.2f        %.2f"%(all_chi2, all_chi2_red))
            print("cut: %.2f        %.2f"%(cut_chi2, cut_chi2_red))

import pylab as plt
import numpy as np
import pickle
from jlu.gc.gcwork import starset
from jlu.util import dataUtil
import pdb

# work_dir = '/Users/jlu/work/ao/ao_optimization/test_stf2_2015_07_08/align_new/'
# plot_dir = work_dir + 'plots/'
plot_dir = 'plots/'
    
def save_starset_to_pickle(align_root, n_lists=12, n_good=9):
    s = starset.StarSet(align_root)

    x_all = s.getArrayFromAllEpochs('xpix')
    y_all = s.getArrayFromAllEpochs('ypix')
    m_all = s.getArrayFromAllEpochs('mag')
    xe_all = s.getArrayFromAllEpochs('xpixerr_a')
    ye_all = s.getArrayFromAllEpochs('ypixerr_a')

    x_old_tmp = x_all[0:n_lists, :]
    y_old_tmp = y_all[0:n_lists, :]
    m_old_tmp = m_all[0:n_lists, :]
    x_new_tmp = x_all[n_lists:, :]
    y_new_tmp = y_all[n_lists:, :]
    m_new_tmp = m_all[n_lists:, :]

    xe_old_tmp = xe_all[0:n_lists, :]
    ye_old_tmp = ye_all[0:n_lists, :]
    xe_new_tmp = xe_all[n_lists:, :]
    ye_new_tmp = ye_all[n_lists:, :]

    x_old = np.ma.masked_where((x_old_tmp < -900), x_old_tmp)
    y_old = np.ma.masked_where((x_old_tmp < -900), y_old_tmp)
    m_old = np.ma.masked_where((x_old_tmp < -900), m_old_tmp)
    x_new = np.ma.masked_where((x_new_tmp < -900), x_new_tmp)
    y_new = np.ma.masked_where((x_new_tmp < -900), y_new_tmp)
    m_new = np.ma.masked_where((x_new_tmp < -900), m_new_tmp)
    
    xe_old = np.ma.masked_where((x_old_tmp < -900), xe_old_tmp)
    ye_old = np.ma.masked_where((x_old_tmp < -900), ye_old_tmp)
    xe_new = np.ma.masked_where((x_new_tmp < -900), xe_new_tmp)
    ye_new = np.ma.masked_where((x_new_tmp < -900), ye_new_tmp)
    
    cnt_old = n_lists - x_old.mask.sum(axis=0)
    cnt_new = n_lists - x_new.mask.sum(axis=0)

    idx = np.where((cnt_old >= n_good) & (cnt_new >= n_good))[0]
    cnt_old = cnt_old[idx]
    cnt_new = cnt_new[idx]

    x_old = x_old[:, idx]
    y_old = y_old[:, idx]
    m_old = m_old[:, idx]
    x_new = x_new[:, idx]
    y_new = y_new[:, idx]
    m_new = m_new[:, idx]

    xe_old = xe_old[:, idx]
    ye_old = ye_old[:, idx]
    xe_new = xe_new[:, idx]
    ye_new = ye_new[:, idx]

    xm_old = x_old.mean(axis=0)
    ym_old = y_old.mean(axis=0)
    mm_old = m_old.mean(axis=0)
    xm_new = x_new.mean(axis=0)
    ym_new = y_new.mean(axis=0)
    mm_new = m_new.mean(axis=0)

    xs_old = x_old.std(axis=0)
    ys_old = y_old.std(axis=0)
    ms_old = m_old.std(axis=0)
    xs_new = x_new.std(axis=0)
    ys_new = y_new.std(axis=0)
    ms_new = m_new.std(axis=0)

    _out = open(align_root + '.pickle', 'w')
    pickle.dump(x_old, _out)
    pickle.dump(y_old, _out)
    pickle.dump(m_old, _out)
    pickle.dump(x_new, _out)
    pickle.dump(y_new, _out)
    pickle.dump(m_new, _out)
    
    pickle.dump(xe_old, _out)
    pickle.dump(ye_old, _out)
    pickle.dump(xe_new, _out)
    pickle.dump(ye_new, _out)

    pickle.dump(xm_old, _out)
    pickle.dump(ym_old, _out)
    pickle.dump(mm_old, _out)
    pickle.dump(xm_new, _out)
    pickle.dump(ym_new, _out)
    pickle.dump(mm_new, _out)

    pickle.dump(xs_old, _out)
    pickle.dump(ys_old, _out)
    pickle.dump(ms_old, _out)
    pickle.dump(xs_new, _out)
    pickle.dump(ys_new, _out)
    pickle.dump(ms_new, _out)

    pickle.dump(cnt_old, _out)
    pickle.dump(cnt_new, _out)

    _out.close()

    return

def load_starset_pickle(align_root, n_lists=12, n_good=9):
    _out = open(align_root + '.pickle', 'r')

    d = dataUtil.DataHolder()
    
    d.x_old = pickle.load(_out)
    d.y_old = pickle.load(_out)
    d.m_old = pickle.load(_out)
    d.x_new = pickle.load(_out)
    d.y_new = pickle.load(_out)
    d.m_new = pickle.load(_out)

    d.xe_old = pickle.load(_out)
    d.ye_old = pickle.load(_out)
    d.xe_new = pickle.load(_out)
    d.ye_new = pickle.load(_out)

    d.xm_old = pickle.load(_out)
    d.ym_old = pickle.load(_out)
    d.mm_old = pickle.load(_out)
    d.xm_new = pickle.load(_out)
    d.ym_new = pickle.load(_out)
    d.mm_new = pickle.load(_out)

    d.xs_old = pickle.load(_out)
    d.ys_old = pickle.load(_out)
    d.ms_old = pickle.load(_out)
    d.xs_new = pickle.load(_out)
    d.ys_new = pickle.load(_out)
    d.ms_new = pickle.load(_out)
    
    d.cnt_old = pickle.load(_out)
    d.cnt_new = pickle.load(_out)

    return d

    
def plot_old_vs_new(align_root, n_lists=12, n_good=9):
    d = load_starset_pickle(align_root, n_lists=n_lists, n_good=n_good)

    # Plot the alignment errors for the old and new in both X and Y
    plt.figure(1)
    plt.clf()
    f, (ax1, ax2) = plt.subplots(2, sharex=True, num=1)

    scale = 9.995 # mas / pixel
    bins = np.arange(0, 0.1*scale, 0.005*scale)

    idx = np.where(d.mm_new < 15)[0]
    
    ax1.hist(d.xe_new[:,idx].compressed()*scale, histtype='step', color='red',
             label='X new', bins=bins, normed=True)
    ax1.hist(d.xe_old[:,idx].compressed()*scale, histtype='step', color='blue',
             label='X old', bins=bins, normed=True)
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Probability')
    
    ax2.hist(d.ye_new[:,idx].compressed()*scale, histtype='step', color='red',
             label='Y new', bins=bins, normed=True)
    ax2.hist(d.ye_old[:,idx].compressed()*scale, histtype='step', color='blue',
             label='Y old', bins=bins, normed=True)
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('align err (mas)')
    
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.savefig(plot_dir + align_root + '_plot_hist_align_err.png')

    # Look at the differences between the old/new alignment errors for each epoch.
    xe_new_mean = d.xe_new[:,idx].mean(axis=1) * scale
    ye_new_mean = d.ye_new[:,idx].mean(axis=1) * scale
    xe_old_mean = d.xe_old[:,idx].mean(axis=1) * scale
    ye_old_mean = d.ye_old[:,idx].mean(axis=1) * scale

    dxe = xe_new_mean - xe_old_mean
    dye = ye_new_mean - ye_old_mean

    plt.clf()
    foo = np.arange(n_lists)
    plt.plot(foo, xe_old_mean, 'bo', label='X Old')
    plt.plot(foo, xe_new_mean, 'ro', label='X New')
    plt.plot(foo, ye_old_mean, 'bs', label='Y Old')
    plt.plot(foo, ye_new_mean, 'rs', label='Y New')
    plt.legend(numpoints=1)
    plt.xlabel('Image Number')
    plt.ylabel('Align Error (mas)')
    plt.ylim(0, 1)
    plt.savefig(plot_dir + align_root + '_plot_align_err.png')

    plt.figure(2)    
    plt.clf()
    foo = np.arange(n_lists)
    plt.plot(foo, dxe, 'ko', label='X')
    plt.plot(foo, dye, 'k^', label='Y')
    plt.ylabel('New - Old Align Error (mas)')
    plt.xlabel('Image Number')
    plt.legend()
    plt.savefig(plot_dir + align_root + '_plot_diff_align_err.png')

    fmt = 'Mean alignment error for {0:s}: Xerr = {1:6.3f} mas   Yerr = {2:6.3f} mas'
    print fmt.format('New', xe_new_mean.mean(), ye_new_mean.mean())
    print fmt.format('Old', xe_old_mean.mean(), ye_old_mean.mean())

    fmt = 'New - Old {0:s} alignment error: mean = {1:6.3f} +/- {2:5.3f} mas   std = {3:6.3f} mas'
    print fmt.format('X', dxe.mean(), dxe.std() / np.sqrt(len(dxe)), dxe.std())
    print fmt.format('Y', dye.mean(), dye.std() / np.sqrt(len(dye)), dye.std())

    # Change in mean positions from old to new. 
    # Map of change of mean positions
    dx = (d.xm_new - d.xm_old) * scale
    dy = (d.ym_new - d.ym_old) * scale
    dr = np.hypot(dx, dy)
    idx = np.where((d.mm_old < 14.5) & (dr < 1))[0]

    # Get rid of mean offset?
    dx -= dx[idx].mean()
    dy -= dy[idx].mean()
    dr = np.hypot(dx, dy)
    idx = np.where((d.mm_old < 14.5) & (dr < 1))[0]

    plt.clf()
    q = plt.quiver(d.xm_old[idx], d.ym_old[idx], dx[idx], dy[idx], scale=5)
    plt.quiver([1050, 0], [100, 0], [0.5, 0], [0.0, 0], scale=8, color='red')
    plt.quiverkey(q, 0.9, 0.1, 0.5, '0.5 mas', labelcolor='red')
    plt.axis('equal')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.title('New - Old Mean Pos (K<14.5)')
    plt.savefig(plot_dir + align_root + '_plot_diff_pos_quiver.png')

    plt.clf()
    bins = np.arange(0, 3, 0.1)
    plt.hist(dr[idx], histtype='step', bins=bins)
    plt.xlabel('New - Old Positions (mas)')
    plt.ylabel('Number of Stars')
    plt.title('Kp < 14.5, dr < 3')
    plt.savefig(plot_dir + align_root + '_plot_diff_pos_hist.png')
    
    good = np.where((d.mm_old < 14.5) & (dr < 1))[0]

    print 'Mean positional difference (Kp<14.5): {0:6.3f} mas'.format(float(dr[idx].mean()))
    print 'Median positional difference (Kp<14.5): {0:6.3f} mas'.format(float(np.median(dr[idx])))
    print 'STD of positional difference (Kp<14.5): {0:6.3f} mas'.format(float(dr[idx].std()))

    # Look at RMS errors. How siginificant are the differences?
    plt.figure(1)
    plt.clf()
    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, num=1)
    ax1.plot(d.mm_new, d.xs_new * scale, 'r.', ms=3, label='New')
    ax1.plot(d.mm_old, d.xs_old * scale, 'b.', ms=3, label='Old')
    ax1.legend(numpoints=1)
    ax1.set_ylabel('X Pos Err (mas)')
    ax1.set_title('RMS Error on Position')

    ax2.plot(d.mm_new, d.ys_new * scale, 'r.', ms=3, label='New')
    ax2.plot(d.mm_old, d.ys_old * scale, 'b.', ms=3, label='Old')
    ax2.set_xlabel('Magnitude')
    ax2.set_ylabel('Y Pos Err (mas)')
    ax2.set_ylim(0, 1)
    plt.savefig(plot_dir + align_root + '_plot_pos_err.png')

    # Only "good" stars
    plt.figure(1)
    plt.clf()
    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, num=1)
    ax1.plot(d.mm_new[good], d.xs_new[good] * scale, 'r.', ms=3, label='New')
    ax1.plot(d.mm_old[good], d.xs_old[good] * scale, 'b.', ms=3, label='Old')
    ax1.legend(numpoints=1)
    ax1.set_ylabel('X Pos Err (mas)')
    ax1.set_title('RMS Error on Position')

    ax2.plot(d.mm_new[good], d.ys_new[good] * scale, 'r.', ms=3, label='New')
    ax2.plot(d.mm_old[good], d.ys_old[good] * scale, 'b.', ms=3, label='Old')
    ax2.set_xlabel('Magnitude')
    ax2.set_ylabel('Y Pos Err (mas)')
    ax2.set_ylim(0, 1)
    plt.savefig(plot_dir + align_root + '_plot_pos_err_good.png')

        
    xeom_new = d.xs_new / np.sqrt(d.cnt_new)
    yeom_new = d.ys_new / np.sqrt(d.cnt_new)
    xeom_old = d.xs_old / np.sqrt(d.cnt_old)
    yeom_old = d.ys_old / np.sqrt(d.cnt_old)

    # Look at EOM errors. How siginificant are the differences?
    plt.figure(1)
    plt.clf()
    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, num=1)
    ax1.plot(d.mm_new, xeom_new * scale, 'r.', ms=3, label='New')
    ax1.plot(d.mm_old, xeom_old * scale, 'b.', ms=3, label='Old')
    ax1.legend(numpoints=1)
    ax1.set_ylabel('X Pos Err (mas)')
    ax1.set_title('EOM Error on Position')

    ax2.plot(d.mm_new, yeom_new * scale, 'r.', ms=3, label='New')
    ax2.plot(d.mm_old, yeom_old * scale, 'b.', ms=3, label='Old')
    ax2.set_xlabel('Magnitude')
    ax2.set_ylabel('Y Pos Err (mas)')
    ax2.set_ylim(0, 1)
    plt.savefig(plot_dir + align_root + '_plot_pos_err_eom.png')

    dm = d.mm_new - d.mm_old
    dms = (d.ms_new - d.ms_old)
    plt.clf()
    plt.plot(d.mm_new, dm, 'k.')
    plt.xlabel('Magnitude')
    plt.ylabel('New - Old Magnitude (mag)')
    plt.savefig(plot_dir + align_root + '_plot_mag_diff.png')

    plt.clf()
    plt.plot(d.mm_new, dms, 'k.')
    plt.xlabel('Magnitude')
    plt.ylabel('New - Old RMS Error on Mag (mag)')
    plt.ylim(-0.1, 0.1)
    plt.savefig(plot_dir + align_root + '_plot_mag_err_diff.png')

    idx = np.where(np.abs(dm) > 
    
    dxs = (d.xs_new - d.xs_old) * scale
    dys = (d.ys_new - d.ys_old) * scale
    plt.clf()
    plt.plot(d.mm_new, dxs, 'ko', label='X', mfc='green', ms=4, mec=None, alpha=0.7)
    plt.plot(d.mm_new, dys, 'k^', label='Y', mfc='purple', ms=4, mec=None, alpha=0.7)
    plt.legend(numpoints=1)
    plt.xlabel('Magnitude')
    plt.ylabel('New - Old RMS Error (mas)')
    plt.ylim(-0.5, 0.5)
    plt.axhline(0, linestyle='--', color='black')
    plt.savefig(plot_dir + align_root + '_plot_diff_rms_err.png')

    # Fix to remove outliers    
    idx1 = np.where((d.mm_new < 14) & (np.abs(dxs) < 1) & (np.abs(dys < 1)))[0]
    idx2 = np.where((d.mm_new < 15) & (np.abs(dxs) < 1) & (np.abs(dys < 1)))[0]
    
    # idx1 = np.where(d.mm_new < 14)[0]
    # idx2 = np.where(d.mm_new < 15)[0]
    
    fmt = 'Mean positional RMS error for {0:s} (K < {1:2d}): Xerr = {2:6.3f} mas   Yerr = {3:6.3f} mas'
    print fmt.format('New', 14, d.xs_new[idx1].mean() * scale, d.ys_new[idx1].mean() * scale)
    print fmt.format('Old', 14, d.xs_old[idx1].mean() * scale, d.ys_old[idx1].mean() * scale)
    print fmt.format('New', 15, d.xs_new[idx2].mean() * scale, d.ys_new[idx2].mean() * scale)
    print fmt.format('Old', 15, d.xs_old[idx2].mean() * scale, d.ys_old[idx2].mean() * scale)

    fmt = 'Mean positional EOM for {0:s} (K < {1:2d}): Xerr = {2:6.3f} mas   Yerr = {3:6.3f} mas'
    print fmt.format('New', 14, xeom_new[idx1].mean() * scale, yeom_new[idx1].mean() * scale)
    print fmt.format('Old', 14, xeom_old[idx1].mean() * scale, yeom_old[idx1].mean() * scale)
    print fmt.format('New', 15, xeom_new[idx2].mean() * scale, yeom_new[idx2].mean() * scale)
    print fmt.format('Old', 15, xeom_old[idx2].mean() * scale, yeom_old[idx2].mean() * scale)

    fmt = 'New - Old {0:1s} RMS Error (Kp < {1:2d}):'
    fmt += '  mean = {2:6.3f} +/- {4:5.3f} mas   std = {3:5.3f} mas   Nstars = {5:4d}'
    print fmt.format('X', 14, dxs[idx1].mean(), dxs[idx1].std(),
                     dxs[idx1].std() / np.sqrt(len(idx1)), len(idx1))
    print fmt.format('X', 15, dxs[idx2].mean(), dxs[idx2].std(),
                     dxs[idx2].std() / np.sqrt(len(idx2)), len(idx2))
    print fmt.format('Y', 14, dys[idx1].mean(), dys[idx1].std(),
                     dys[idx1].std() / np.sqrt(len(idx1)), len(idx1))
    print fmt.format('Y', 15, dys[idx2].mean(), dys[idx2].std(),
                     dys[idx2].std() / np.sqrt(len(idx2)), len(idx2))

    plt.clf()
    bins = np.arange(-1, 1, 0.05)
    plt.hist(dxs[idx1], histtype='step', color='green', label='X', bins=bins)
    plt.hist(dys[idx1], histtype='step', color='purple', label='Y', bins=bins)
    plt.xlabel('New - Old RMS Error (mas)')
    plt.ylabel('Number of Stars')
    plt.axvline(0, linestyle='--', color='black')
    plt.legend()
    plt.title('Kp < 14')
    plt.savefig(plot_dir + align_root + '_plot_diff_rms_err_hist_14.png')
    
    plt.clf()
    plt.hist(dxs[idx2], histtype='step', color='green', label='X', bins=bins)
    plt.hist(dys[idx2], histtype='step', color='purple', label='Y', bins=bins)
    plt.xlabel('New - Old RMS Error (mas)')
    plt.ylabel('Number of Stars')
    plt.axvline(0, linestyle='--', color='black')
    plt.legend()
    plt.title('Kp < 15')
    plt.savefig(plot_dir + align_root + '_plot_diff_rms_err_hist_15.png')

    plt.clf()
    plt.plot()

    plt.figure(3, figsize=(14, 6))
    plt.clf()
    plt.subplots_adjust(left=0.08)
    f, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, shareY=True, num=3)
    ptx = ax1.scatter(d.xm_new[idx2], d.ym_new[idx2], c=dxs[idx2], s=60, vmin=-0.5, vmax=0.5, edgecolor='')
    ax1.set_title('X RMS Err Diff')
    ax1.set_xlabel('X Pos (pix)')
    ax1.set_ylabel('Y Pos (pix)')
    ax1.axis('equal')
    ax1.set_xlim(0, 1100)
    ax1.set_ylim(0, 1100)
    pty = ax2.scatter(d.xm_new[idx2], d.ym_new[idx2], c=dys[idx2], s=60, vmin=-0.5, vmax=0.5, edgecolor='')
    ax2.set_title('Y RMS Err Diff')
    ax2.set_xlabel('X Pos (pix)')
    ax2.axis('equal')
    ax2.set_xlim(0, 1100)
    ax2.set_ylim(0, 1100)
    plt.colorbar(ptx, ax=ax1, label='New - Old RMS Error (mas)')
    plt.colorbar(pty, ax=ax2, label='New - Old RMS Error (mas)')
    plt.savefig(plot_dir + align_root + '_plot_diff_rms_err_map_15.png')



    return
    

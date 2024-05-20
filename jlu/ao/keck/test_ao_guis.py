import PyQt5
import pylab as plt
from matplotlib.patches import Rectangle, Circle, Arrow, Ellipse
import numpy as np

def plot_pcu_schematic(x, y, z, rot):
    fig = plt.figure(1, figsize=(10, 6))
    plt.clf()
    ax1 = fig.add_axes([0.08, 0.08, 0.55, 0.85])
    ax2 = fig.add_axes([0.70, 0.08, 0.32, 0.85])

    #####
    # Draw the X stage.
    #####
    x_wh = np.array([461, 70]) # mm measured from L412 CAD
    x_cxy = np.array([x_wh[0]/2.0, x_wh[1]/2.0])
    x_xy = x_cxy - x_wh / 2.0

    xstage = Rectangle(x_xy, x_wh[0], x_wh[1], ec='black', fc='grey')

    # Label
    x_cxy = x_xy + x_wh/2.0
    ax1.add_patch(xstage)
    ax1.annotate(f'X={x}', x_cxy, color='w',
                weight='bold', fontsize=12, ha='center', va='center')

    #####
    # Draw the Y stage.
    #####
    y_wh = np.array([70, 361])
    y_cxy = np.array([x, x_wh[1] + y_wh[1]/2.0])
    y_xy = y_cxy - y_wh / 2.0
    ystage = Rectangle(y_xy, y_wh[0], y_wh[1], ec='black', fc='grey') 

    # Label
    ax1.add_patch(ystage)
    ax1.annotate(f'Y={y}', y_cxy, color='w', rotation=90,
                weight='bold', fontsize=12, ha='center', va='center')

    #####
    # Draw the Z stage.
    # Make the Z fiber and pinhole stages. They ride together.
    #####
    z_cxy = np.array([x - 40, y_xy[1] + y])

    ### Pinhole
    # Width/Height of entire rectangle, not centered on pinhole mask.
    zp_wh  = np.array([169, 110])
    # center of hole in, true center in y
    zp_cxy = np.array([y_cxy[0] + y_wh[0]/2.0 + 105, y_xy[1] + y + 186])
    # radius of hole
    zp_r = 23 # mm
    # lower-left corner of rectangle
    zp_xy  = np.array([y_cxy[0] + y_wh[0]/2.0, zp_cxy[1] - zp_wh[1]/2.0])
    
    zstage_p = Rectangle(zp_xy, zp_wh[0], zp_wh[1], ec='black', fc='grey')
    zstage_pc = Circle(zp_cxy, zp_r)

    ax1.add_patch(zstage_p)
    ax1.add_patch(zstage_pc)
    
    ### Fiber
    # Width/Height of entire rectangle, not centered on fiber.
    zf_wh = np.array([62, 20])
    # center of fiber in x, true center in y
    zf_cxy = np.array([y_cxy[0] + y_wh[0]/2.0 + 54, y_xy[1] + y + 115])
    # radius of fiber chuck.. approximate.
    zf_r = 7
    # lower-left corner of rectangle
    zf_xy = np.array([y_cxy[0] + y_wh[0]/2.0, zf_cxy[1] - zf_wh[1]/2.0])

    zstage_f = Rectangle(zf_xy, zf_wh[0], zf_wh[1], ec='black', fc='grey')
    zstage_fc = Circle(zf_cxy, zf_r)
    
    ax1.add_patch(zstage_f)
    ax1.add_patch(zstage_fc)
    
    ### KPF fold mirror
    # Width/Height of entire rectangle, not centered on fiber.
    zk_wh = np.array([127 + 25, 78])
    # center of fold mirror in x and y, true center in y
    zk_cxy = np.array([y_cxy[0] + y_wh[0]/2.0 + 127, y_xy[1] + y + 49])
    # radius of fold mirror. approximate.
    zk_r = 22.5 # mm
    # lower-left corner of rectangle
    zk_xy = np.array([y_cxy[0] + y_wh[0]/2.0, zk_cxy[1] - zk_wh[1]/2.0])

    zstage_k = Rectangle(zk_xy, zk_wh[0], zk_wh[1], ec='black', fc='grey')
    zstage_kc = Circle(zk_cxy, zk_r)
    
    ax1.add_patch(zstage_k)
    ax1.add_patch(zstage_kc)
    
    
    # Label
    ax1.annotate('Fiber', zf_xy, color='w',
                weight='bold', fontsize=12, ha='left', va='bottom')
    ax1.annotate('Pinhole', zp_cxy, color='w',
                weight='bold', fontsize=12, ha='center', va='center')
    ax1.annotate('KPF Fold', [zk_xy[0] + 30, zk_cxy[1]], color='w',
                weight='bold', fontsize=12, ha='left', va='center')

    #####
    # Plot Optical Axis
    #####
    opt_cxy = np.array([y_wh[0]/2.0 + 54 + 150, zf_cxy[1]])
    ax1.plot(opt_cxy[0], opt_cxy[1], 'r*', ms=20)
    
    ax1.set_aspect('equal')
    ax1.set_xlim([0, 500])
    ax1.set_ylim([0, 500])
    ax1.set_title('Looking from telescope')
    ax1_b = ax1.secondary_yaxis('right', functions=(lambda y: y - x_wh[1], lambda Y: Y + x_wh[1]))
    ax1_b.set_ylabel('Y Motor Coords')
    ax1.set_ylabel('Bench Coords')

    ax2.set_aspect('equal')
    ax2.set_xlim([0, 200])
    ax2.set_ylim([0, 500])
    ax2.set_title('From KPF FIU')
    
    return
    

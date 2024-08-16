from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import brewer2mpl
import matplotlib
import netCDF4 as nc
import numpy as np
import os

def force_aspect(ax,aspect=1):
    """Force plot aspects."""

    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def load_python_data(model_name):
    """Load data from model of interest. Input model name as string."""

    path = os.getcwd() + '/model_data/' + model_name + '.npz'
    model = np.load(path,allow_pickle=True)

    conc_day = model['conc_day']
    conc_month = model['conc_month']

    year = 2
    start = (year-1)*12
    end = start + 12
    conc_month = conc_month[:,:,start:end]
    return conc_month, conc_day


def nrmse(check,comp):
    """Calculate normalized root mean square error.
    Root mean square error normalized by the difference between maximum and minimum concentration of the check field."""

    avg = np.zeros(7)
    dif = np.zeros(7)
    rms = np.zeros(7)
    for i in range(0,7):
        avg[i] = np.abs(np.mean(check[i,:,:]))
        dif[i] = np.max(check[i,:,:]) - np.min(check[i,:,:])
        rms[i] = np.power( np.mean( np.power( check[i,:,:]-comp[i,:,:], 2 ) )   ,0.5)
    nrmse = 100*rms/avg    
    # nrmse = 100*rms/dif

    return nrmse


def plot_bfm1(check,comp,model_name):
    """Create plots of oxygen concentration for BFM1 vs BFM50 (check and comp)."""
    
    # ---------------------------------------------------------------------------------------------------------------------------------
    # Plot Style
    plt.rc('font', family='serif', size=20)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', labelsize=20, linewidth=1)
    # ---------------------------------------------------------------------------------------------------------------------------------
    # Legend Default
    plt.rc('legend', framealpha=1.0, facecolor='white', frameon=True, edgecolor='black')
    # ---------------------------------------------------------------------------------------------------------------------------------
    # Plot Colors
    bmap = brewer2mpl.get_map('Paired', 'qualitative', 10)
    colors = bmap.mpl_colors
    # ---------------------------------------------------------------------------------------------------------------------------------
    # Colorbar Limits
    clow   = 180
    chigh  = 235
    # ---------------------------------------------------------------------------------------------------------------------------------
    # Oxygen Plots
    fig,axes = plt.subplots(1,2,figsize=[12,5])
    plt.subplot(1,2,1)
    plt.imshow(check[1,:,:],extent=[0,12,150,0],aspect='auto',cmap='jet')
    ax = plt.gca()
    plt.title('(a)')
    plt.xlabel('Month',fontsize=14)
    plt.xticks([0.5,2.5,4.5,6.5,8.5,10.5], ['J','M','M','J','S','N'])
    plt.ylabel('Depth (m)',fontsize=14)
    plt.yticks([0,50,100,150])
    plt.clim(clow,chigh)
    force_aspect(ax,aspect=1)

    plt.subplot(1,2,2)
    plt.imshow(comp[1,:,:],extent=[0,12,150,0],aspect='auto',cmap='jet')
    ax = plt.gca()
    plt.title('(b)')
    plt.xlabel('Month',fontsize=14)
    plt.xticks([0.5,2.5,4.5,6.5,8.5,10.5], ['J','M','M','J','S','N'])
    plt.yticks([0,50,100,150],[])
    plt.colorbar(orientation='vertical')
    plt.clim(clow,chigh)
    force_aspect(ax,aspect=1)

    plt.tight_layout()
    fig.subplots_adjust(wspace=-0.5,hspace=0.3)

    plt.savefig(model_name + '.jpg')


def plot_fields(check,comp,model_name):
    """Create plots of concentration fields for the two input models (check and comp)."""
    
    # ---------------------------------------------------------------------------------------------------------------------------------
    # Plot Style
    plt.rc('font', family='serif', size=20)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', labelsize=20, linewidth=1)
    # ---------------------------------------------------------------------------------------------------------------------------------
    # Legend Default
    plt.rc('legend', framealpha=1.0, facecolor='white', frameon=True, edgecolor='black')
    # ---------------------------------------------------------------------------------------------------------------------------------
    # Plot Colors
    bmap = brewer2mpl.get_map('Paired', 'qualitative', 10)
    colors = bmap.mpl_colors
    # ---------------------------------------------------------------------------------------------------------------------------------
    # Titles
    title_check = ['(a) Chl-a','(b) Oxygen','(c) Nitrate','(d) Phosphate','(e) PON','(f) NPP','(g) DIC']
    title_comp = ['(h) Chl-a','(i) Oxygen','(j) Nitrate','(k) Phosphate','(l) PON','(m) NPP','(n) DIC']
    title = ['(a) Chl-a','(b) Oxygen','(c) Nitrate','(d) Phosphate','(e) Chl-a','(f) Oxygen','(g) Nitrate','(h) Phoshate','(i) PON','(j) NPP','(k) DIC','(l) PON','(m) NPP','(n) DIC']
    # ---------------------------------------------------------------------------------------------------------------------------------
    # Colorbar Limits
    clow   = [0,180,0,0,0.1,0,30]
    chigh  = [0.225,235,2.5,0.075,0.405,2.0,170]

    # ---------------------------------------------------------------------------------------------------------------------------------
    # Field Plots
       
    fig,axes = plt.subplots(4,4,figsize=[16,15])
    for i in range(0,7):
        plt.subplot(4,4,i+1)
        plt.imshow(check[i,:,:],extent=[0,12,150,0],aspect='auto',cmap='jet')
        ax = plt.gca()
        plt.xticks([0.5,2.5,4.5,6.5,8.5,10.5], ['J','M','M','J','S','N'])
        plt.xlabel('Month',fontsize=14)
        if i%4 == 0:
            plt.yticks([0,50,100,150])
            plt.ylabel('Depth (m)',fontsize=14)
        else:
            plt.yticks([0,50,100,150],[])
        plt.title(title_check[i],fontsize=20)
        plt.clim(clow[i],chigh[i])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)

    for i in range(7,14):
        plt.subplot(4,4,i+2)
        plt.imshow(comp[i-7,:,:],extent=[0,12,150,0],aspect='auto',cmap='jet')
        ax = plt.gca()
        plt.xticks([0.5,2.5,4.5,6.5,8.5,10.5], ['J','M','M','J','S','N'])
        plt.xlabel('Month',fontsize=14)
        plt.yticks([0,50,100,150])
        if i%4 == 3:
            plt.yticks([0,50,100,150])
            plt.ylabel('Depth (m)',fontsize=14)
        else:
            plt.yticks([0,50,100,150],[])
        plt.title(title_comp[i-7],fontsize=20)
        plt.clim(clow[i-7],chigh[i-7]) 
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)   

    fig.delaxes(axes[1,3])
    fig.delaxes(axes[3,3])

    plt.tight_layout(h_pad=0.75, w_pad=0.75)

    plt.savefig(model_name + '.jpg')

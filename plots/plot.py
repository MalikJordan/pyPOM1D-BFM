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


def load_fortran_data(model_name):
    """Load fortran bfm56-pom1d data."""

    path = os.getcwd() + '/model_data/' + model_name + '.nc'
    variables = nc.Dataset(path)
    variables = variables.variables

    # Extract fields of interest
    chlorophyll = variables['Chla'][:]
    oxygen = variables['O2o'][:]
    nitrate = variables['N3n'][:]
    phosphate = variables['N1p'][:]
    # pon = variables['P1n'][:] + variables['P2n'][:] + variables['P3n'][:] + variables['P4n'][:] + variables['Z3n'][:] + variables['Z4n'][:] + variables['Z5n'][:] + variables['Z6n'][:] \
    #     + variables['R1n'][:] + variables['R6n'][:]
    pon = variables['R6n'][:] + variables['P1n'][:] + variables['P2n'][:] + variables['P3n'][:] + variables['P4n'][:]
    production = (variables['ruPTc'][:] - variables['resPP'][:] - variables['resZT'][:])/12
    dic = (variables['DIC'][:])*(variables['ERHO'][:])*(12/1000)

    # Write as array
    chlorophyll = np.asarray(chlorophyll)
    oxygen = np.asarray(oxygen)
    nitrate = np.asarray(nitrate)
    phosphate = np.asarray(phosphate)
    pon = np.asarray(pon)
    production = np.asarray(production)
    dic = np.asarray(dic)

    # Transpose array
    chlorophyll = chlorophyll.transpose()
    oxygen = oxygen.transpose()
    nitrate = nitrate.transpose()
    phosphate = phosphate.transpose()
    pon = pon.transpose()
    production = production.transpose()
    dic = dic.transpose()

    # Load data into matrix
    data_fortran = np.zeros((7,chlorophyll.shape[0],chlorophyll.shape[1]))
    data_fortran[0,:,:] = chlorophyll
    data_fortran[1,:,:] = oxygen
    data_fortran[2,:,:] = nitrate
    data_fortran[3,:,:] = phosphate
    data_fortran[4,:,:] = pon
    data_fortran[5,:,:] = production
    data_fortran[6,:,:] = dic

    # Calculate monthly averages for year 2 of simulation
    avg_data_fortran = np.zeros((7,150,12))
    for spec in range(0,7):
        for year in range(1,2):
        # for year in range(14,15):
            for month in range(0,12):
                for day in range(0,30):
                    avg_data_fortran[spec,:,month] = avg_data_fortran[spec,:,month] + data_fortran[spec,:,(day + (month*30) + (year*360))]
    avg_data_fortran = avg_data_fortran/30

    return avg_data_fortran, data_fortran


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
    title_check = ['(a) Chl-a','(b) Oxygen','(c) Nitrate','(d) Phoshate','(e) PON','(f) NPP','(g) DIC']
    title_comp = ['(h) Chl-a','(i) Oxygen','(j) Nitrate','(k) Phoshate','(l) PON','(m) NPP','(n) DIC']
    title = ['(a) Chl-a','(b) Oxygen','(c) Nitrate','(d) Phoshate','(e) Chl-a','(f) Oxygen','(g) Nitrate','(h) Phoshate','(i) PON','(j) NPP','(k) DIC','(l) PON','(m) NPP','(n) DIC']
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


model_name = 'pyPOM1D-BFM50'
fortran_model='bfm56_pom1d'
pyPOM50, pyPOM50_daily = load_python_data(model_name)
bfm50, bfm50_daily = load_fortran_data(fortran_model)

# ---------------------------------------------------------------------------------------------------------------------------------
# NRMSE
nrmse_pyPOM50 = nrmse(bfm50,pyPOM50)
species = ['Chl-a','Oxygen','Nitrate','Phosphate','PON','NPP','DIC']
print('NRMSE (%) - pyPOM50 vs bfm50')
for i in range(0,7):
    print(species[i], ' - ', nrmse_pyPOM50[i])
print()

# ---------------------------------------------------------------------------------------------------------------------------------
# Plots
plot_fields(pyPOM50,bfm50,model_name) # pyPOM50 set as 'check' for plotting to remain consistent with placement in other plots


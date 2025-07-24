from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import brewer2mpl
import matplotlib
import netCDF4 as nc
import numpy as np
import os
import colormaps as cmaps

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


def load_fortran_data_bfm17(model_name):
    """Load fortran bfm56-pom1d data."""

    path = os.getcwd() + '/model_data/' + model_name + '.nc'
    variables = nc.Dataset(path)
    variables = variables.variables

    # Extract fields of interest
    chlorophyll = variables['Chla'][:]
    oxygen = variables['O2o'][:]
    nitrate = variables['N3n'][:]
    phosphate = variables['N1p'][:]
    pon = variables['R6n'][:] + variables['P2n'][:]
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


def load_python_data_15yr(model_name):
    """Load data from model of interest. Input model name as string."""

    path = os.getcwd() + '/model_data/' + model_name + '.npz'
    model = np.load(path,allow_pickle=True)

    concentration = model['conc']
    chlorophyll = model['chl']
    production = model['npp']

    # Load data into matrix
    # data_python = np.zeros((7,150,730))
    data_python = np.zeros((7,chlorophyll.shape[0],chlorophyll.shape[1]))
    data_python[0,:,:] = chlorophyll   # Chlorophyll-a
    data_python[1,:,:] = concentration[:,0,:]  # Oxygen
    data_python[2,:,:] = concentration[:,2,:]  # Nitrate
    data_python[3,:,:] = concentration[:,1,:]  # Phosphate
    # data_python[4,:,:] = concentration[:,11,:] + concentration[:,16,:] + concentration[:,20,:] + concentration[:,24,:] + concentration[:,28,:] \
    #                 + concentration[:,31,:] + concentration[:,34,:] + concentration[:,37,:] + concentration[:,40,:] + concentration[:,45,:] # Particulate Organic Nitrogen
    data_python[4,:,:] = concentration[:,45,:] + concentration[:,11,:] + concentration[:,16,:] + concentration[:,20,:] + concentration[:,24,:]
    data_python[5,:,:] = production    # Net Primary Production
    data_python[6,:,:] = concentration[:,48,:] # Dissolved Inorganic Carbon

    # Calculate monthly averages for year 2 of simulation
    avg_data_python = np.zeros((7,150,12))
    for spec in range(0,7):
        for year in range(1,2):
        # for year in range(14,15):
            for month in range(0,12):
                for day in range(0,30):
                    avg_data_python[spec,:,month] = avg_data_python[spec,:,month] + data_python[spec,:,(day + (month*30) + (year*360))]
    avg_data_python = avg_data_python/30

    return avg_data_python, data_python


def nrmse(check,comp,type):
    """Calculate normalized root mean square error.
    Root mean square error normalized by the mean concentration of the check field."""

    avg = np.zeros(7)
    dif = np.zeros(7)
    rms = np.zeros(7)
    std = np.zeros(7)

    for i in range(0,7):
        avg[i] = np.abs(np.mean(check[i,:,:]))
        dif[i] = np.max(check[i,:,:]) - np.min(check[i,:,:])
        rms[i] = np.power( np.mean( np.power( check[i,:,:]-comp[i,:,:], 2 ) )   ,0.5)
        std[i] = np.std(check[i,:,:])
    
    if type == "avg":      nrmse = 100*rms/avg    
    elif type == "dif":    nrmse = 100*rms/dif
    elif type == "std":    nrmse = 100*rms/std

    return nrmse


def line_plots(check,comp):
    
    plt.rc('font', family='serif', size=16)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    iters = check.shape[2]
    x = np.linspace(0,iters-1,iters)
    # Chl-a
    depth = [0,29,59,89,119,149]
    tit = ['0m','30m','60m','90m','120m','150m']
    plot_titles = ['(a)','(b)','(c)','(d)','(e)','(f)']
    fig,axes = plt.subplots(2,3,figsize=[20,10])
    for i in range(0,6):
        plt.subplot(2,3,i+1)
        plt.plot(x,check[0,depth[i],:],'-')
        plt.plot(x,comp[0,depth[i],:],':')
        ax = plt.gca()
        plt.title(plot_titles[i])
        plt.grid(linestyle = '--', linewidth = 0.5)
        if i<3:
            plt.xlabel('')
            plt.xticks([360,720,1080,1440,1800,2160,2520,2880,3240,3600,3960,4320,4680,5040,5400],['','','','','','','','','','','','','','',''])
        else:
            plt.xlabel('Year')
            plt.xticks([360,720,1080,1440,1800,2160,2520,2880,3240,3600,3960,4320,4680,5040,5400],['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
        
        plt.xlim(0,iters)
        # plt.legend(['Python','Fortran'])
    
    # plt.suptitle('Chlorophyll-a')
    plt.tight_layout()
    fig_name = 'chl_line_plots.jpg'
    plt.savefig(fig_name)
    
    # Oxygen
    fig,axes = plt.subplots(2,3,figsize=[20,10])
    for i in range(0,6):
        plt.subplot(2,3,i+1)
        plt.plot(x,check[1,depth[i],:],'-')
        plt.plot(x,comp[1,depth[i],:],':')
        ax = plt.gca()
        plt.title(plot_titles[i])
        plt.grid(linestyle = '--', linewidth = 0.5)
        if i<3:
            plt.xlabel('')
            plt.xticks([360,720,1080,1440,1800,2160,2520,2880,3240,3600,3960,4320,4680,5040,5400],['','','','','','','','','','','','','','',''])
        else:
            plt.xlabel('Year')
            plt.xticks([360,720,1080,1440,1800,2160,2520,2880,3240,3600,3960,4320,4680,5040,5400],['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
        
        plt.xlim(0,iters)

    # plt.suptitle('Oxygen')
    plt.tight_layout()
    fig_name = 'o2o_line_plots.jpg'
    plt.savefig(fig_name)

    # Nitrate
    fig,axes = plt.subplots(2,3,figsize=[20,10])
    for i in range(0,6):
        plt.subplot(2,3,i+1)
        plt.plot(x,check[2,depth[i],:],'-')
        plt.plot(x,comp[2,depth[i],:],':')
        ax = plt.gca()
        plt.title(plot_titles[i])
        plt.grid(linestyle = '--', linewidth = 0.5)
        if i<3:
            plt.xlabel('')
            plt.xticks([360,720,1080,1440,1800,2160,2520,2880,3240,3600,3960,4320,4680,5040,5400],['','','','','','','','','','','','','','',''])
        else:
            plt.xlabel('Year')
            plt.xticks([360,720,1080,1440,1800,2160,2520,2880,3240,3600,3960,4320,4680,5040,5400],['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
        
        plt.xlim(0,iters)

    # plt.suptitle('Nitrate')
    plt.tight_layout()
    fig_name = 'n3n_line_plots.jpg'
    plt.savefig(fig_name)

    # Phosphate
    fig,axes = plt.subplots(2,3,figsize=[20,10])
    for i in range(0,6):
        plt.subplot(2,3,i+1)
        plt.plot(x,check[3,depth[i],:],'-')
        plt.plot(x,comp[3,depth[i],:],':')
        ax = plt.gca()
        plt.title(plot_titles[i])
        plt.grid(linestyle = '--', linewidth = 0.5)
        if i<3:
            plt.xlabel('')
            plt.xticks([360,720,1080,1440,1800,2160,2520,2880,3240,3600,3960,4320,4680,5040,5400],['','','','','','','','','','','','','','',''])
        else:
            plt.xlabel('Year')
            plt.xticks([360,720,1080,1440,1800,2160,2520,2880,3240,3600,3960,4320,4680,5040,5400],['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
        
        plt.xlim(0,iters)

    # plt.suptitle('Phosphate')
    plt.tight_layout()
    fig_name = 'n1p_line_plots.jpg'
    plt.savefig(fig_name)

    # PON
    fig,axes = plt.subplots(2,3,figsize=[20,10])
    for i in range(0,6):
        plt.subplot(2,3,i+1)
        plt.plot(x,check[4,depth[i],:],'-')
        plt.plot(x,comp[4,depth[i],:],':')
        ax = plt.gca()
        plt.title(plot_titles[i])
        plt.grid(linestyle = '--', linewidth = 0.5)
        if i<3:
            plt.xlabel('')
            plt.xticks([360,720,1080,1440,1800,2160,2520,2880,3240,3600,3960,4320,4680,5040,5400],['','','','','','','','','','','','','','',''])
        else:
            plt.xlabel('Year')
            plt.xticks([360,720,1080,1440,1800,2160,2520,2880,3240,3600,3960,4320,4680,5040,5400],['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
        
        plt.xlim(0,iters)

    # plt.suptitle('Particulate Organic Nitrogen')
    plt.tight_layout()
    fig_name = 'pon_line_plots.jpg'
    plt.savefig(fig_name)

    # NPP
    fig,axes = plt.subplots(2,3,figsize=[20,10])
    for i in range(0,6):
        plt.subplot(2,3,i+1)
        plt.plot(x,check[5,depth[i],:],'-')
        plt.plot(x,comp[5,depth[i],:],':')
        ax = plt.gca()
        plt.title(plot_titles[i])
        plt.grid(linestyle = '--', linewidth = 0.5)
        if i<3:
            plt.xlabel('')
            plt.xticks([360,720,1080,1440,1800,2160,2520,2880,3240,3600,3960,4320,4680,5040,5400],['','','','','','','','','','','','','','',''])
        else:
            plt.xlabel('Year')
            plt.xticks([360,720,1080,1440,1800,2160,2520,2880,3240,3600,3960,4320,4680,5040,5400],['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
        
        plt.xlim(0,iters)

    # plt.suptitle('Net Primary Production')
    plt.tight_layout()
    fig_name = 'npp_line_plots.jpg'
    plt.savefig(fig_name)

    # DIC
    fig,axes = plt.subplots(2,3,figsize=[20,10])
    for i in range(0,6):
        plt.subplot(2,3,i+1)
        plt.plot(x,check[6,depth[i],:],'-')
        plt.plot(x,comp[6,depth[i],:],':')
        ax = plt.gca()
        plt.title(plot_titles[i])
        plt.grid(linestyle = '--', linewidth = 0.5)
        if i<3:
            plt.xlabel('')
            plt.xticks([360,720,1080,1440,1800,2160,2520,2880,3240,3600,3960,4320,4680,5040,5400],['','','','','','','','','','','','','','',''])
        else:
            plt.xlabel('Year')
            plt.xticks([360,720,1080,1440,1800,2160,2520,2880,3240,3600,3960,4320,4680,5040,5400],['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
        
        plt.xlim(0,iters)

    # plt.suptitle('Dissolved Inorganic Carbon')
    plt.tight_layout()
    fig_name = 'o3c_line_plots.jpg'
    plt.savefig(fig_name)




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
    plt.imshow(check[1,:,:],extent=[0,12,150,0],aspect='auto',cmap=cmaps.viridis)
    ax = plt.gca()
    plt.title('(a)')
    plt.xlabel('Month',fontsize=14)
    plt.xticks([0.5,2.5,4.5,6.5,8.5,10.5], ['J','M','M','J','S','N'])
    plt.ylabel('Depth (m)',fontsize=14)
    plt.yticks([0,50,100,150])
    plt.clim(clow,chigh)
    force_aspect(ax,aspect=1)

    plt.subplot(1,2,2)
    plt.imshow(comp[1,:,:],extent=[0,12,150,0],aspect='auto',cmap=cmaps.viridis)
    ax = plt.gca()
    plt.title('(b)')
    plt.xlabel('Month',fontsize=14)
    plt.xticks([0.5,2.5,4.5,6.5,8.5,10.5], ['J','M','M','J','S','N'])
    plt.yticks([0,50,100,150],[])
    cbar = plt.colorbar(orientation='vertical')
    cbar.ax.get_yaxis().labelpad = 25
    cbar.ax.set_ylabel('mmol O m$^{-3}$', fontsize=14, rotation=270)
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
        plt.imshow(check[i,:,:],extent=[0,12,150,0],aspect='auto',cmap=cmaps.viridis)
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
        plt.imshow(comp[i-7,:,:],extent=[0,12,150,0],aspect='auto',cmap=cmaps.viridis)
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

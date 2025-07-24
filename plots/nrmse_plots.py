import numpy as np
from functions import *
from matplotlib import pyplot as plt

# ---------------------------------------------------------------------------------------------------------------------------------
# Data
model50 = 'pyPOM1D-BFM50'
pyPOM50, pyPOM50_daily = load_python_data(model50)

model36 = 'pyPOM1D-BFM36'
pyPOM36, pyPOM36_daily = load_python_data(model36)

model35 = 'pyPOM1D-BFM35'
pyPOM35, pyPOM35_daily = load_python_data(model35)

model34 = 'pyPOM1D-BFM34'
pyPOM34, pyPOM34_daily = load_python_data(model34)

model23 = 'pyPOM1D-BFM23'
pyPOM23, pyPOM23_daily = load_python_data(model23)

model1  = 'pyPOM1D-BFM1'
pyPOM1,  pyPOM1_daily  = load_python_data(model1)

bfm50 = 'bfm56_pom1d'
bfm50, bfm50_daily = load_fortran_data(bfm50)

bfm17 = 'bfm17_pom1d'
bfm17, bfm17_daily = load_fortran_data_bfm17(bfm17)


# ---------------------------------------------------------------------------------------------------------------------------------
# Calculate NRMSE
# ---------------------------------------------------------------------------------------------------------------------------------
nrmse_pyPOM36_mean = nrmse(pyPOM50,pyPOM36,'avg')
nrmse_pyPOM36_std = nrmse(pyPOM50,pyPOM36,'std')

nrmse_pyPOM35_mean  = nrmse(pyPOM50,pyPOM35,'avg')
nrmse_pyPOM35_std  = nrmse(pyPOM50,pyPOM35,'std')
nrmse_pyPOM35_mean[-1] = float('nan')    # DIC field not included in bfm35
nrmse_pyPOM35_std[-1]  = float('nan')    # DIC field not included in bfm35


nrmse_pyPOM34_mean  = nrmse(pyPOM50,pyPOM34,'avg')
nrmse_pyPOM34_std = nrmse(pyPOM50,pyPOM34,'std')
nrmse_pyPOM34_mean[-1] = float('nan')    # DIC field not included in bfm34
nrmse_pyPOM34_std[-1]  = float('nan')    # DIC field not included in bfm34


nrmse_pyPOM23_mean  = nrmse(pyPOM50,pyPOM23,'avg')
nrmse_pyPOM23_std = nrmse(pyPOM50,pyPOM23,'std')

nrmse_bfm17_mean  = nrmse(bfm50,bfm17,'avg')
nrmse_bfm17_std   = nrmse(bfm50,bfm17,'std')
nrmse_bfm17_mean[-1] = float('nan')    # DIC field not included in bfm17
nrmse_bfm17_std[-1]  = float('nan')    # DIC field not included in bfm17


# ---------------------------------------------------------------------------------------------------------------------------------
# Plots
# poster_plots(pyPOM50,pyPOM36,pyPOM23,pyPOM1,bfm17)

# ---------------------------------------------------------------------------------------------------------------------------------
# NRMSE -34, -35, -36
# ---------------------------------------------------------------------------------------------------------------------------------
# Create dictionary for group
models = ("pyPOM34","pyPOM35","pyPOM36")
errors = {
    'Chl-a': [nrmse_pyPOM34_mean[0], nrmse_pyPOM35_mean[0], nrmse_pyPOM36_mean[0]],
    'Oxygen': [nrmse_pyPOM34_mean[1], nrmse_pyPOM35_mean[1], nrmse_pyPOM36_mean[1]],
    'Nitrate': [nrmse_pyPOM34_mean[2], nrmse_pyPOM35_mean[2], nrmse_pyPOM36_mean[2]],
    'Phosphate': [nrmse_pyPOM34_mean[3], nrmse_pyPOM35_mean[3], nrmse_pyPOM36_mean[3]],
    'PON': [nrmse_pyPOM34_mean[4], nrmse_pyPOM35_mean[4], nrmse_pyPOM36_mean[4]],
    'NPP': [nrmse_pyPOM34_mean[5], nrmse_pyPOM35_mean[5], nrmse_pyPOM36_mean[5]],
    'DIC': [nrmse_pyPOM34_mean[6], nrmse_pyPOM35_mean[6], nrmse_pyPOM36_mean[6]]
}

x = 1.5*np.arange(len(models))  # Label locations
bar_width = 0.2
multiplier = 0

fig, (ax2,ax1) = plt.subplots(2, 1, sharex=True, figsize=(10,7), gridspec_kw={'height_ratios': [1, 7]})
fig.subplots_adjust(left=0.1, hspace=0.1)
for field, error in errors.items():
    offset = bar_width * multiplier
    if field == 'Nitrate':  
        rects1 = ax1.bar(x + offset, [0.16,error[1],error[2]], bar_width, label=field)
        rects1[0].set_clip_on(False)
    else:
        rects1 = ax1.bar(x + offset, error, bar_width, label=field)
    # rects1 = ax1.bar(x + offset, error, bar_width, label=field)
    rects2 = ax2.bar(x + offset, error, bar_width, label=field)
    # if field == 'Nitrate':  #rects_nit = rects
        # ax1.bar_label(rects, labels=[round(nrmse_pyPOM34[2],3),'',''], label_type="center", padding=-17, rotation=90, fontsize=16)
        # rects1[0].set_clip_on(False)
    # ax.bar_label(rects, label_type="center", rotation=90, fontsize=14)
    multiplier += 1

# Get handles and labels for the legend
handles, labels = ax1.get_legend_handles_labels()

# Add a single legend to the figure
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.735, 0.88), fontsize=12)

# Turn off spines between subplots
ax1.spines.top.set_visible(False)
ax2.spines.bottom.set_visible(False)

# Turn off x-axis ticks
ax1.xaxis.set_ticks_position('none') 
ax2.xaxis.set_ticks_position('none') 

# Set axis limits
ax1.set_ylim([0,0.15])
ax2.set_ylim([1.2195,1.24])

# Label axes
fig.supylabel('NRMSE [%]', fontsize=12)
ax2.set_yticks([1.22,1.24])
ax1.set_xticks([x[0]+2.5*bar_width, x[1]+2.5*bar_width, x[2]+3*bar_width], models, fontsize=12)     # Center model names

# Turn on grid lines
ax1.grid(axis='y')
ax2.grid(axis='y')
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [1, 1], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)

plt.savefig("plots/figures/nrmse1_mean.jpg")



# Create dictionary for group
models = ("pyPOM34","pyPOM35","pyPOM36")
errors = {
    'Chl-a': [nrmse_pyPOM34_std[0], nrmse_pyPOM35_std[0], nrmse_pyPOM36_std[0]],
    'Oxygen': [nrmse_pyPOM34_std[1], nrmse_pyPOM35_std[1], nrmse_pyPOM36_std[1]],
    'Nitrate': [nrmse_pyPOM34_std[2], nrmse_pyPOM35_std[2], nrmse_pyPOM36_std[2]],
    'Phosphate': [nrmse_pyPOM34_std[3], nrmse_pyPOM35_std[3], nrmse_pyPOM36_std[3]],
    'PON': [nrmse_pyPOM34_std[4], nrmse_pyPOM35_std[4], nrmse_pyPOM36_std[4]],
    'NPP': [nrmse_pyPOM34_std[5], nrmse_pyPOM35_std[5], nrmse_pyPOM36_std[5]],
    'DIC': [nrmse_pyPOM34_std[6], nrmse_pyPOM35_std[6], nrmse_pyPOM36_std[6]]
}

fig, (ax3,ax2,ax1) = plt.subplots(3, 1, sharex=True, figsize=(15,9), gridspec_kw={'height_ratios': [1,1,1]})
fig.subplots_adjust(left=0.075, hspace=0.2)
multiplier = 0  # Reset multiplier

for field, error in errors.items():
    offset = bar_width * multiplier

    if field == 'Oxygen':
        rects1 = ax1.bar(x + offset, [0.12,0.12,0.12], bar_width, label=field)
        [rects1[i].set_clip_on(False) for i in range(3)]
        rects2 = ax2.bar(x + offset, error, bar_width, label=field)
        rects3 = ax3.bar(x + offset, error, bar_width, label=field)

    elif field == 'Nitrate':  
        rects1 = ax1.bar(x + offset, [0.12,error[1],error[2]], bar_width, label=field)
        rects1[0].set_clip_on(False)
        rects2 = ax2.bar(x + offset, error, bar_width, label=field)
        rects3 = ax3.bar(x + offset, error, bar_width, label=field)

    elif field == 'PON':  
        rects1 = ax1.bar(x + offset, [0.07,error[1],error[2]], bar_width, label=field)
        rects1[0].set_clip_on(False)
        rects2 = ax2.bar(x + offset, error, bar_width, label=field)
        rects3 = ax3.bar(x + offset, error, bar_width, label=field)

    else:
        rects1 = ax1.bar(x + offset, error, bar_width, label=field)
        rects2 = ax2.bar(x + offset, error, bar_width, label=field)
        rects3 = ax3.bar(x + offset, error, bar_width, label=field)

    multiplier += 1

# Get handles and labels for the legend
handles, labels = ax1.get_legend_handles_labels()

# Add a single legend to the figure
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.75, 0.88), fontsize=16)

# Turn off spines between subplots
ax1.spines.top.set_visible(False)
ax2.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax3.spines.bottom.set_visible(False)

# Turn off x-axis ticks
ax1.xaxis.set_ticks_position('none') 
ax2.xaxis.set_ticks_position('none') 
ax3.xaxis.set_ticks_position('none') 

# Set axis limits
ax1.set_ylim([0,0.05])
ax2.set_ylim([0.2495,0.3])
ax3.set_ylim([0.795,1.3])

# Label axes
fig.supylabel('NRMSE [%]', fontsize=18)
ax1.set_xticks([x[0]+2.5*bar_width, x[1]+2.5*bar_width, x[2]+3*bar_width], models, fontsize=18)     # Center model names

# Turn on grid lines
ax1.grid(axis='y')
ax2.grid(axis='y')
ax3.grid(axis='y')
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)
ax3.set_axisbelow(True)

# Add dashed
d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [1, 1], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
ax3.plot([0, 1], [0, 0], transform=ax3.transAxes, **kwargs)

plt.savefig("plots/figures/nrmse1_std.jpg")

# ---------------------------------------------------------------------------------------------------------------------------------
# NRMSE -17, -23
# ---------------------------------------------------------------------------------------------------------------------------------
# Create dictionary for group
models = ("BFM17","pyPOM23")
errors = {
    'Chl-a': [nrmse_bfm17_mean[0], nrmse_pyPOM23_mean[0]],
    'Oxygen': [nrmse_bfm17_mean[1], nrmse_pyPOM23_mean[1]],
    'Nitrate': [nrmse_bfm17_mean[2], nrmse_pyPOM23_mean[2]],
    'Phosphate': [nrmse_bfm17_mean[3], nrmse_pyPOM23_mean[3]],
    'PON': [nrmse_bfm17_mean[4], nrmse_pyPOM23_mean[4]],
    'NPP': [nrmse_bfm17_mean[5], nrmse_pyPOM23_mean[5]],
    'DIC': [nrmse_bfm17_mean[6], nrmse_pyPOM23_mean[6]]
}

x = 0.6*np.arange(len(models))  # Label locations
bar_width = 0.075
multiplier = 0

fig, ax = plt.subplots(layout="tight",figsize=(10,8))

for field, error in errors.items():
    offset = bar_width * multiplier
    rects = ax.bar(x + offset, error, bar_width, label=field)
    # ax.bar_label(rects, padding=3)
    multiplier += 1


# Turn off x-axis ticks
ax.xaxis.set_ticks_position('none') 

# Add labels
ax.set_ylabel('NRMSE [%]', fontsize=16, labelpad=20)
ax.set_ylim([0,180])
ax.set_xticks([x[0]+2.5*bar_width,x[1]+3*bar_width], models, fontsize=16)
ax.grid(axis='y')
ax.set_axisbelow(True)
ax.legend(fontsize=16)
plt.savefig("plots/figures/nrmse2_mean.jpg")



models = ("BFM17","pyPOM23")
errors = {
    'Chl-a': [nrmse_bfm17_std[0], nrmse_pyPOM23_std[0]],
    'Oxygen': [nrmse_bfm17_std[1], nrmse_pyPOM23_std[1]],
    'Nitrate': [nrmse_bfm17_std[2], nrmse_pyPOM23_std[2]],
    'Phosphate': [nrmse_bfm17_std[3], nrmse_pyPOM23_std[3]],
    'PON': [nrmse_bfm17_std[4], nrmse_pyPOM23_std[4]],
    'NPP': [nrmse_bfm17_std[5], nrmse_pyPOM23_std[5]],
    'DIC': [nrmse_bfm17_std[6], nrmse_pyPOM23_std[6]]
}

x = 0.6*np.arange(len(models))  # Label locations
bar_width = 0.075
multiplier = 0

fig, ax = plt.subplots(layout="tight",figsize=(10,8))

for field, error in errors.items():
    offset = bar_width * multiplier
    rects = ax.bar(x + offset, error, bar_width, label=field)
    # ax.bar_label(rects, padding=3)
    multiplier += 1


# Turn off x-axis ticks
ax.xaxis.set_ticks_position('none') 

# Add labels
ax.set_ylabel('NRMSE [%]', fontsize=16, labelpad=20)
ax.set_ylim([0,200])
ax.set_xticks([x[0]+2.5*bar_width,x[1]+3*bar_width], models, fontsize=16)
ax.grid(axis='y')
ax.set_axisbelow(True)
ax.legend(fontsize=14)
plt.savefig("plots/figures/nrmse2_std.jpg")


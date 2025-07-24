import numpy as np
from functions import *

# ---------------------------------------------------------------------------------------------------------------------------------
# Data
model_name='pyPOM1D-BFM50-15yr'
fortran_model='bfm56_pom1d-15yr'
pyPOM50, pyPOM50_daily = load_python_data_15yr(model_name)
bfm50, bfm50_daily = load_fortran_data(fortran_model)

# ---------------------------------------------------------------------------------------------------------------------------------
# NRMSE
nrmse_pyPOM50 = nrmse(bfm50,pyPOM50,'std')
species = ['Chl-a','Oxygen','Nitrate','Phosphate','PON','NPP','DIC']
print('NRMSE (%) - pyPOM50 vs bfm50')
for i in range(0,7):
    print(species[i], ' - ', nrmse_pyPOM50[i])
print()

# ---------------------------------------------------------------------------------------------------------------------------------
# Plots
model_name='pyPOM1D-BFM50'
plot_fields(pyPOM50,bfm50,model_name) # pyPOM50 set as 'check' for plotting to remain consistent with placement in other plots
line_plots(pyPOM50_daily,bfm50_daily) # pyPOM50 set as 'check' for plotting to remain consistent with placement in other plots
import numpy as np
from functions import *

# ---------------------------------------------------------------------------------------------------------------------------------
# Data
model_name='pyPOM1D-BFM23'
full_model='pyPOM1D-BFM50'
bfm17_data='bfm17_pom1d'
bfm50_data='bfm56_pom1d'
pyPOM23, pyPOM23_daily = load_python_data(model_name)
pyPOM50, pyPOM50_daily = load_python_data(full_model)

bfm17, bfm17_daily = load_fortran_data_bfm17(bfm17_data)
bfm50, bfm50_daily = load_fortran_data(bfm50_data)
# ---------------------------------------------------------------------------------------------------------------------------------
# NRMSE
nrmse_pyPOM23 = nrmse(pyPOM50,pyPOM23,'std')
species = ['Chl-a','Oxygen','Nitrate','Phosphate','PON','NPP','DIC']
print()
print('NRMSE (%) - pyPOM23 vs pyPOM50')
for i in range(0,7):
    print(species[i], ' - ', nrmse_pyPOM23[i])
print()

# nrmse_bfm17 = nrmse(bfm50,bfm17)
# species = ['Chl-a','Oxygen','Nitrate','Phosphate','PON','NPP','DIC']
# print()
# print('NRMSE (%) - BFM17 vs BFM50')
# for i in range(0,7):
#     print(species[i], ' - ', nrmse_bfm17[i])
# print()

# ---------------------------------------------------------------------------------------------------------------------------------
# Plots
plot_fields(pyPOM50,pyPOM23,model_name) # pyPOM50 set as 'check' for plotting to remain consistent with placement in other plots
plot_fields(bfm17,pyPOM23,'BFM17-BFM23') # autom
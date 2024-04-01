import numpy as np
from functions import *

# ---------------------------------------------------------------------------------------------------------------------------------
# Data
model_name='pyPOM1D-BFM34'
full_model='pyPOM1D-BFM50'
pyPOM34, pyPOM34_daily = load_python_data(model_name)
pyPOM50, pyPOM50_daily = load_python_data(full_model)

# ---------------------------------------------------------------------------------------------------------------------------------
# NRMSE
nrmse_pyPOM34 = nrmse(pyPOM50,pyPOM34)
species = ['Chl-a','Oxygen','Nitrate','Phosphate','PON','NPP','DIC']
print('NRMSE (%) - pyPOM34 vs pyPOM50')
for i in range(0,7):
    print(species[i], ' - ', nrmse_pyPOM34[i])
print()

# ---------------------------------------------------------------------------------------------------------------------------------
# Plots
plot_fields(pyPOM50,pyPOM34,model_name) # pyPOM50 set as 'check' for plotting to remain consistent with placement in other plots


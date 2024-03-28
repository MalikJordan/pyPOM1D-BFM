import numpy as np
import os
from bfm.data_classes import BfmPhysicalVariables, OutputAverages
from pom.calculations import density_profile, kinetic_energy_profile, meridional_velocity_profile, temperature_and_salinity_profiles, zonal_velocity_profile
from pom.data_classes import Diffusion, ForcingManagerCounters, KineticEnergyAndVelocity, MonthlyForcing, Nutrients, Stresses, TemperatureAndSalinity, VerticalGrid
from pom.forcing import forcing_manager
from pom.initialization import read_temperature_and_salinity_initial_coditions
from pom.parameters import PomBfm
from pom_bfm.coupling import light_distribution, pom_bfm_1d, pom_to_bfm, vertical_extinction
from pom_bfm.initialization import initialize_bfm_in_pom

current_path = os.getcwd()

# Import Constant Parameters
pom_bfm_parameters = PomBfm()

# General Initialization
diffusion = Diffusion(pom_bfm_parameters.vertical_layers)

kinetic_energy = KineticEnergyAndVelocity(1.e-07,pom_bfm_parameters.vertical_layers)
kinetic_energy_times_length = KineticEnergyAndVelocity(1.e-07,pom_bfm_parameters.vertical_layers)
zonal_velocity = KineticEnergyAndVelocity(0.,pom_bfm_parameters.vertical_layers)
meridional_velocity = KineticEnergyAndVelocity(0.,pom_bfm_parameters.vertical_layers)

nutrients = Nutrients()

wind_stress = Stresses()
bottom_stress = Stresses()

temperature = TemperatureAndSalinity(pom_bfm_parameters.vertical_layers)
salinity = TemperatureAndSalinity(pom_bfm_parameters.vertical_layers)

vertical_grid = VerticalGrid(pom_bfm_parameters.vertical_layers)
vertical_grid.coordinate_system(pom_bfm_parameters.kl1,pom_bfm_parameters.kl2,pom_bfm_parameters.vertical_layers)

# Read Temperature and Salinity Inital Conditions
if pom_bfm_parameters.ihotst == 0:
    t0 = 0.
    temperature.current, temperature.backward, salinity.current, salinity.backward = read_temperature_and_salinity_initial_coditions(pom_bfm_parameters.vertical_layers)
    density = density_profile(pom_bfm_parameters,temperature,salinity,vertical_grid)
elif pom_bfm_parameters.ihotst == 1:
    # get_rst()
    pass

if not pom_bfm_parameters.pom_only:
    # Initialize BFM
    d3state,d3stateb = initialize_bfm_in_pom(vertical_grid)
    bfm_phys_vars = BfmPhysicalVariables(pom_bfm_parameters.num_boxes)
    d3ave = OutputAverages(pom_bfm_parameters.days,pom_bfm_parameters.months,pom_bfm_parameters.num_boxes)
else:
    bfm_phys_vars = BfmPhysicalVariables()

# Create output file
model_name = 'pyPOM1D-BFM' + str(d3state.shape[1])
print('---------------------------')
print('- Model -->',model_name,'-')
print('---------------------------')

output_file = open('output_file.txt','w')
output_file.write('---------------------------\n')
output_file.write('- Model --> ')
output_file.write(model_name)
output_file.write(' -\n')
output_file.write('---------------------------\n\n')
output_file.close()

# Begin the time march
counters = ForcingManagerCounters()
month1_data = MonthlyForcing(pom_bfm_parameters.vertical_layers)
month2_data = MonthlyForcing(pom_bfm_parameters.vertical_layers)

for i in range(0,pom_bfm_parameters.iterations_needed):
    t = t0 + (pom_bfm_parameters.dti * i * pom_bfm_parameters.dayi)

    # Turbulence closure
    kinetic_energy.forward[:] = kinetic_energy.backward[:]
    kinetic_energy_times_length.forward[:] = kinetic_energy_times_length.backward[:]

    kinetic_energy, kinetic_energy_times_length, diffusion, vertical_grid = kinetic_energy_profile(pom_bfm_parameters,vertical_grid, diffusion, density, zonal_velocity, meridional_velocity, 
                                                                                                    kinetic_energy, kinetic_energy_times_length, wind_stress, bottom_stress)
    
    # Define all forcings
    temperature.forward, temperature.interpolated, salinity.forward, salinity.interpolated, \
        shortwave_radiation, temperature.surface_flux, wind_stress, bfm_phys_vars.wgen, bfm_phys_vars.weddy, \
        month1_data, month2_data, counters, nutrients, inorganic_suspended_matter = forcing_manager(i,counters,month1_data,month2_data,pom_bfm_parameters)
    
    # Temperature and salinity computation
    if pom_bfm_parameters.idiagn == 0:
        # Prognostic mode
        # Temperature and salinity fully computed by model
        temperature.surface_value = temperature.forward[0]
        salinity.surface_value = salinity.forward[0]

        if pom_bfm_parameters.trt != 0:
            for j in range(0, pom_bfm_parameters.vertical_layers):
                if (-vertical_grid.vertical_coordinates_staggered[j] * pom_bfm_parameters.h) >= pom_bfm_parameters.upperh:
                    temperature.lateral_advection[j] = (temperature.interpolated[j] - temperature.current[j]) / (pom_bfm_parameters.trt * pom_bfm_parameters.sec_per_day)

        if pom_bfm_parameters.srt != 0:
            for j in range(0, pom_bfm_parameters.vertical_layers):
                if (-vertical_grid.vertical_coordinates_staggered[j] * pom_bfm_parameters.h) >= pom_bfm_parameters.upperh:
                    salinity.lateral_advection[j] = (salinity.interpolated[j] - salinity.current[j]) / (pom_bfm_parameters.srt * pom_bfm_parameters.sec_per_day)

        # Calculate surface salinity flux
        salinity.surface_flux = -(salinity.surface_value - salinity.current[0]) * pom_bfm_parameters.srt / pom_bfm_parameters.sec_per_day

        # Calculate temperature
        temperature.forward[:] = temperature.backward[:] + (temperature.lateral_advection[:] * pom_bfm_parameters.dti2)
        temperature = temperature_and_salinity_profiles(pom_bfm_parameters, vertical_grid, diffusion, temperature, shortwave_radiation, 'Temperature')

        # Calculate salinity
        salinity.forward[:] = salinity.backward[:] + (salinity.lateral_advection[:] * pom_bfm_parameters.dti2)
        salinity = temperature_and_salinity_profiles(pom_bfm_parameters, vertical_grid, diffusion, salinity, shortwave_radiation, 'Salinity')

        # Mix the timestep (Asselin filter)
        temperature.current[:] = temperature.current[:] + 0.5 * pom_bfm_parameters.smoth * (temperature.forward[:] + temperature.backward[:] - 2. * temperature.current[:])
        salinity.current[:] = salinity.current[:] + 0.5 * pom_bfm_parameters.smoth * (salinity.forward[:] + salinity.backward[:] - 2. * salinity.current[:])

    zonal_velocity.forward[:] = zonal_velocity.backward[:] + pom_bfm_parameters.dti2 * pom_bfm_parameters.coriolis * meridional_velocity.current[:]
    zonal_velocity, bottom_stress = zonal_velocity_profile(pom_bfm_parameters, vertical_grid, wind_stress, bottom_stress, diffusion, zonal_velocity)

    meridional_velocity.forward[:] = meridional_velocity.backward[:] - pom_bfm_parameters.dti2 * pom_bfm_parameters.coriolis * zonal_velocity.current[:]
    velocity, bottom_stress = meridional_velocity_profile(pom_bfm_parameters, vertical_grid, wind_stress, bottom_stress, diffusion, meridional_velocity)

    # Mix the timestep (Asselin filter)
    kinetic_energy.current[:] = kinetic_energy.current[:] + 0.5 * pom_bfm_parameters.smoth * (kinetic_energy.forward[:] + kinetic_energy.backward[:] - 2. * kinetic_energy.current[:])
    kinetic_energy_times_length.current[:] = kinetic_energy_times_length.current[:] + 0.5 * pom_bfm_parameters.smoth * (kinetic_energy_times_length.forward[:] + kinetic_energy_times_length.backward[:] - 2. * kinetic_energy_times_length.current[:])

    zonal_velocity.current[:] = zonal_velocity.current[:] + 0.5 * pom_bfm_parameters.smoth * (zonal_velocity.forward[:] + zonal_velocity.backward[:] - 2. * zonal_velocity.current[:])
    meridional_velocity.current[:] = meridional_velocity.current[:] + 0.5 * pom_bfm_parameters.smoth * (meridional_velocity.forward[:] + meridional_velocity.backward[:] - 2. * meridional_velocity.current[:])

    # Restore the time sequence
    kinetic_energy.backward[:] = kinetic_energy.current[:]
    kinetic_energy.current[:] = kinetic_energy.forward[:]
    kinetic_energy_times_length.backward[:] = kinetic_energy_times_length.current[:]
    kinetic_energy_times_length.current[:] = kinetic_energy_times_length.forward[:]

    zonal_velocity.backward[:] = zonal_velocity.current[:]
    zonal_velocity.current[:] = zonal_velocity.forward[:]
    meridional_velocity.backward[:] = meridional_velocity.current[:]
    meridional_velocity.current[:] = meridional_velocity.forward[:]

    temperature.backward[:] = temperature.current[:]
    temperature.current[:] = temperature.forward[:]
    salinity.backward[:] = salinity.current[:]
    salinity.current[:] = salinity.forward[:]

    # Update density
    density = density_profile(pom_bfm_parameters,temperature,salinity,vertical_grid)
    
    if not pom_bfm_parameters.pom_only:
        bfm_phys_vars = pom_to_bfm(bfm_phys_vars, vertical_grid, temperature, salinity, inorganic_suspended_matter, shortwave_radiation, density, wind_stress)
        bfm_phys_vars.vertical_extinction = vertical_extinction(bfm_phys_vars, d3state)
        bfm_phys_vars.irradiance = light_distribution(bfm_phys_vars)
        
        d3state, d3stateb, d3ave = pom_bfm_1d(i, vertical_grid, t, diffusion, nutrients, bfm_phys_vars, d3state, d3stateb, d3ave)    

# Write Outputs
np.savez('/model_data' + model_name + '.npz',conc_day=d3ave.daily_ave,conc_month=d3ave.monthly_ave)
print('Main done')
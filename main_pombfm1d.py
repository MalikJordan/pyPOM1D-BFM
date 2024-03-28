import copy
import numpy as np
import os
from bfm.data_classes import BfmPhysicalVariables, OutputAverages
from bfm.parameters import Co2Flux
from pom.calculations import density_profile, kinetic_energy_profile, meridional_velocity_profile, temperature_and_salinity_profiles, zonal_velocity_profile
from pom.data_classes import Diffusion, ForcingManagerCounters, KineticEnergyAndVelocity, MonthlyForcing, Nutrients, Stresses, TemperatureAndSalinity, VerticalGrid
from pom.forcing import forcing_manager
from pom.initialization import read_temperature_and_salinity_initial_coditions
from pom.parameters import PomBfm
from pom_bfm.coupling import light_distribution, pom_bfm_1d, pom_to_bfm, vertical_extinction
from pom_bfm.initialization import initialize_bfm_in_pom
from reduction.included_species import included_species, remove_species
from reduction.modified_DRGEP import modified_DRGEP

current_path = os.getcwd()

# Import Constant Parameters
co2_flux_parameters = Co2Flux()
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

REDUCE_BFM = True
if not pom_bfm_parameters.pom_only:
    # Initialize BFM
    d3state,d3stateb = initialize_bfm_in_pom(vertical_grid)
    bfm_phys_vars = BfmPhysicalVariables(pom_bfm_parameters.num_boxes,co2_flux_parameters.ph_initial)
    d3ave = OutputAverages(pom_bfm_parameters.days,pom_bfm_parameters.months,pom_bfm_parameters.num_boxes)

    species_names = {'O2o':0,  'N1p':1,  'N3n':2,  'N4n':3,  'O4n':4,  'N5s':5,  'N6r':6,  'B1c':7,  'B1n':8,  'B1p':9, 
                     'P1c':10, 'P1n':11, 'P1p':12, 'P1l':13, 'P1s':14, 'P2c':15, 'P2n':16, 'P2p':17, 'P2l':18,
                     'P3c':19, 'P3n':20, 'P3p':21, 'P3l':22, 'P4c':23, 'P4n':24, 'P4p':25, 'P4l':26,
                     'Z3c':27, 'Z3n':28, 'Z3p':29, 'Z4c':20, 'Z4n':31, 'Z4p':32, 'Z5c':33, 'Z5n':34, 'Z5p':35,
                     'Z6c':36, 'Z6n':37, 'Z6p':38, 'R1c':39, 'R1n':40, 'R1p':41, 'R2c':42, 'R3c':43, 'R6c':44, 
                     'R6n':45, 'R6p':46, 'R6s':47, 'O3c':48, 'O3h':49}
    
    if REDUCE_BFM:
        bfm_phys_vars.depth = vertical_grid.vertical_spacing[:-1] * pom_bfm_parameters.h

        # Reduce model
        reduction, error_limit = modified_DRGEP(d3state,bfm_phys_vars)

        error_data = reduction['error_data'][-2:]
        if error_data[-1] > error_limit:
            model = -2
        else:
            model = -1

        species_removed_data = reduction['species_removed_data'][-2:]
        species_removed = species_removed_data[model]

        d3state, d3stateb, species = remove_species(d3state,d3stateb,species_names,species_removed)
        include = included_species(species)

    else: # BFM50
        # Include all species in original BFM50
        include = included_species(species_names)
        species = species_names
else:
    bfm_phys_vars = BfmPhysicalVariables()

# Create output file
num_species = d3state.shape[1]
model_name = 'pyPOM1D-BFM' + str(num_species)
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
    
    # Zero forcing data for removed species
    if not pom_bfm_parameters.pom_only:
        if 'O2o' not in species:  # Oxygen
            nutrients.O2bott = 0
        if 'N1p' not in species:  # Phosphate
            nutrients.PO4bott = 0
            nutrients.PO4surf = 0
        if 'N3n' not in species:  # Nitrate
            nutrients.NO3bott = 0
            nutrients.NO3surf = 0
        if 'N4n' not in species:  # Ammonium
            nutrients.NH4surf = 0
            nutrients.PONbott_grad = 0
        if 'N5s' not in species:  # Silicate
            nutrients.SIO4surf = 0

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
        bfm_phys_vars.vertical_extinction = vertical_extinction(bfm_phys_vars, d3state, species)
        bfm_phys_vars.irradiance = light_distribution(bfm_phys_vars)

        d3state, d3stateb, d3ave = pom_bfm_1d(i, vertical_grid, t, diffusion, nutrients, bfm_phys_vars, d3state, d3stateb, d3ave, include, species)    

# Write Outputs
np.savez('/model_data' + model_name + '.npz',conc_day=d3ave.daily_ave,conc_month=d3ave.monthly_ave)
print('Main done')
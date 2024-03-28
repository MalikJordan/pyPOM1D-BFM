import numpy as np
from bfm.bfm50.bfm50_rate_eqns import bfm50_rate_eqns
from bfm.bfm50.state_variables import state_vars
from bfm.data_classes import BfmStateVariables
from bfm.parameters import Bfm, Constants, Environment, Phyto1, Phyto2, Phyto3, Phyto4
from pom.parameters import PomBfm
from pom.calculations import temperature_and_salinity_profiles
import os

current_path = os.getcwd()
bfm_parameters = Bfm()
constant_parameters = Constants()
env_parameters = Environment()
phyto1_parameters = Phyto1()
phyto2_parameters = Phyto2()
phyto3_parameters = Phyto3()
phyto4_parameters = Phyto4()
pom_bfm_parameters = PomBfm()


def average_fields(d3ave, d3state, chl, npp, case, species):
    """
    Description: Calculates daily averages for fields of interest
    
    Fields: [0] Chlorophyll-a
            [1] Oxygen
            [2] Nitrate
            [3] Phosphate
            [4] Particulate Organic Nitrogen
            [5] Net Primary Production
            [6] Dissolved Inorganic Carbon
    """

    num_boxes = d3state.shape[0]

    o2o, n1p, n3n, n4n, o4n, n5s, n6r, b1c, b1n, b1p, \
    p1c, p1n, p1p, p1l, p1s, p2c, p2n, p2p, p2l, p3c, p3n, p3p, p3l, p4c, p4n, p4p, p4l, \
    z3c, z3n, z3p, z4c, z4n, z4p, z5c, z5n, z5p, z6c, z6n, z6p, \
    r1c, r1n, r1p, r2c, r3c, r6c, r6n, r6p, r6s, o3c, o3h   = state_vars(d3state,num_boxes,species)

    oxy = o2o
    nit = n3n
    phos = n1p
    pon = p1n + p2n + p3n + p4n + z3n + z4n + z5n + z6n + r1n + r6n
    dic = o3c

    if case == 'Accumulate':
        d3ave.daily_ave[0,:,d3ave.day] = d3ave.daily_ave[0,:,d3ave.day] + chl
        d3ave.daily_ave[1,:,d3ave.day] = d3ave.daily_ave[1,:,d3ave.day] + oxy
        d3ave.daily_ave[2,:,d3ave.day] = d3ave.daily_ave[2,:,d3ave.day] + nit
        d3ave.daily_ave[3,:,d3ave.day] = d3ave.daily_ave[3,:,d3ave.day] + phos
        d3ave.daily_ave[4,:,d3ave.day] = d3ave.daily_ave[4,:,d3ave.day] + pon
        d3ave.daily_ave[5,:,d3ave.day] = d3ave.daily_ave[5,:,d3ave.day] + npp
        d3ave.daily_ave[6,:,d3ave.day] = d3ave.daily_ave[6,:,d3ave.day] + dic
    
    elif case == 'Mean':
        # Add values from current timestep
        d3ave.daily_ave[0,:,d3ave.day] = d3ave.daily_ave[0,:,d3ave.day] + chl
        d3ave.daily_ave[1,:,d3ave.day] = d3ave.daily_ave[1,:,d3ave.day] + oxy
        d3ave.daily_ave[2,:,d3ave.day] = d3ave.daily_ave[2,:,d3ave.day] + nit
        d3ave.daily_ave[3,:,d3ave.day] = d3ave.daily_ave[3,:,d3ave.day] + phos
        d3ave.daily_ave[4,:,d3ave.day] = d3ave.daily_ave[4,:,d3ave.day] + pon
        d3ave.daily_ave[5,:,d3ave.day] = d3ave.daily_ave[5,:,d3ave.day] + npp
        d3ave.daily_ave[6,:,d3ave.day] = d3ave.daily_ave[6,:,d3ave.day] + dic

        # Take the average
        d3ave.daily_ave[:,:,d3ave.day] = d3ave.daily_ave[:,:,d3ave.day]/d3ave.count

        # Update monthly average, Take average after 30 days
        d3ave.monthly_ave[:,:,d3ave.month] = d3ave.monthly_ave[:,:,d3ave.month] + d3ave.daily_ave[:,:,d3ave.day]
        if (d3ave.day != 0) & ((d3ave.day + 1) % 30 == 0):
            d3ave.monthly_ave[:,:,d3ave.month] = d3ave.monthly_ave[:,:,d3ave.month]/30
            d3ave.month = d3ave.month + 1   # Move to next month

        # Move to next day
        d3ave.day = d3ave.day + 1

        # Reset counter for next day
        d3ave.count = 0
    
    return d3ave


def detritus_sedimentation():
    """
    Description: Calculates teh sedimentation rate for detritus
    """
    # FROM namelists_bfm
    # p_rR6m        [m/d]   detritus sinking rate
    # p_burvel_R6   [m/d]   Bottom burial velocity for detritus

    p_rR6m = 1.
    p_burvel_R6 = 1.

    # FROM PelGlobal.F90 (145-148)
    detritus_sedimentation_rate = p_rR6m * np.ones(pom_bfm_parameters.num_boxes)

    if not bfm_parameters.pom_bfm:
        detritus_sedimentation_rate[-1] = p_burvel_R6

    return detritus_sedimentation_rate


def light_distribution(bfm_phys_vars):
    """
    Description: Defines the irradiance profile
    """
    # Calcluations taken from bfm/bfm50/Functions/phyto.py - Updated from 0D to 1D
    # Array calculations from CalcLightDistribution.F90

    bfm_phys_vars.irradiance[0] = bfm_phys_vars.irradiance[0]*env_parameters.epsilon_PAR/constant_parameters.e2w
    for i in range(1,pom_bfm_parameters.num_boxes):
        bfm_phys_vars.irradiance[i] = bfm_phys_vars.irradiance[i-1] * np.exp( -1. * bfm_phys_vars.vertical_extinction[i-1] * bfm_phys_vars.depth[i-1])

    return bfm_phys_vars.irradiance


def phyto_sedimentation():
    """
    Description: Calculates the sedimentation rates for the phytoplankton groups
    """
    # From namelists_bfm
    # p_rPIm        [m/d]   phytoplanktom background sinking rate
    # p_burvel_PI   [m/d]   Botttom burial velocity for detritus
    p_rPIm = [0.0, 0.0, 0.0, 0.0]
    p_burvel_PI = 0.0

    num_phyto_groups = 4

    # From PelGLobal.F90 (149-154)
    phyto_sedimentation_rates = np.zeros((pom_bfm_parameters.num_boxes,num_phyto_groups))
    for i in range(0,num_phyto_groups):
        phyto_sedimentation_rates[:,i] = p_rPIm[i]

        if not bfm_parameters.pom_bfm:
            phyto_sedimentation_rates[-1,i] = p_burvel_PI

    return phyto_sedimentation_rates


def pom_bfm_1d(i, vertical_grid, time, diffusion, nutrients, bfm_phys_vars, d3state, d3stateb, d3ave, include, species):
    """
    Description: Handles the calling of all the subroutines that provides BFM with all the needed information about the physical environment
    """
    bfm_rates = np.zeros((pom_bfm_parameters.num_boxes,bfm_parameters.num_d3_box_states))

    dOdt_wind = np.zeros(pom_bfm_parameters.num_boxes)
    do3cdt_air_sea_flux = np.zeros(pom_bfm_parameters.num_boxes)

    bfm_rates, bfm_phys_vars, dOdt_wind, do3cdt_air_sea_flux, chl, npp = bfm50_rate_eqns(i, time, d3state, bfm_phys_vars, include, species, seasonal_cycle=False)

    d3state, d3stateb = vertical_diffusivity(vertical_grid, diffusion, nutrients, d3state, d3stateb, bfm_rates, bfm_phys_vars, dOdt_wind, do3cdt_air_sea_flux, include, species)

    d3ave.count += 1    # count initialized to 0, update before averaging
    if (d3ave.count > 0) and (d3ave.count < (pom_bfm_parameters.sec_per_day/pom_bfm_parameters.dti)):
        d3ave = average_fields(d3ave, d3state, chl, npp, 'Accumulate', species)
    elif d3ave.count == (pom_bfm_parameters.sec_per_day/pom_bfm_parameters.dti):
        d3ave = average_fields(d3ave, d3state, chl, npp, 'Mean', species)

    return d3state, d3stateb, d3ave


def pom_to_bfm(bfm_phys_vars, vertical_grid, temperature, salinity, inorganic_suspended_matter, shortwave_radiation, vertical_density_profile, wind_stress):
    """
    Description: Passes the physical variables to the BFM

    :return: seawater density, temperature and salinity, suspended sediment load,
             photosynthetically available radiation, gridpoint depth, and wind speed
    """

    #   1D arrays for BFM
    bfm_phys_vars.temperature = temperature.backward[:-1]
    bfm_phys_vars.salinity = salinity.backward[:-1]
    bfm_phys_vars.density = (vertical_density_profile[:-1] * 1.e03) + 1.e03
    bfm_phys_vars.suspended_matter = inorganic_suspended_matter
    bfm_phys_vars.depth = vertical_grid.vertical_spacing[:-1] * pom_bfm_parameters.h

    bfm_phys_vars.irradiance[0] = -1. * shortwave_radiation * pom_bfm_parameters.water_specific_heat_times_density

    wind = np.sqrt(wind_stress.zonal**2 + wind_stress.meridional**2) * 1.e03
    bfm_phys_vars.wind = np.sqrt(wind/(1.25 * 0.0014))

    return bfm_phys_vars


def vertical_advection(property, sinking_velocity, vertical_grid):
    """"
    Description: Handles the sinking of BFM state variablles. Sinking is treated as downward vertical advection
                 computed with upstream finite differences.
    NOTE: Downward velocities are negative
    """
    # sinking velocity input from vdiff_SOS
    property.current[-1] = property.current[-2]
    property.backward[-1] = property.backward[-2]


    property.forward[0] = vertical_grid.vertical_spacing_reciprocal[0] * property.current[0] * sinking_velocity[1]
    for i in range(1,pom_bfm_parameters.vertical_layers-1):
        property.forward[i] = vertical_grid.vertical_spacing_reciprocal[i] * (property.current[i] * sinking_velocity[i + 1] - property.current[i - 1] * sinking_velocity[i])

    return property


def vertical_diffusivity(vertical_grid, diffusion, nutrients, d3state, d3stateb, bfm_rates, bfm_phys_vars, dOdt_wind, do3cdt_air_sea_flux, include, species):
    """
    Description: Calculates the vertical diffusivity of BFM biochemical components and
                 integrats BFM state variables with Source Splitting (SoS) method
    """

    # The input general cir. vertical vel. is suppose to be in m/s
    W_ON = 1.0

    # The input eddy vertical vel. is provided in m/d
    Weddy_ON = 0.1/86400.0  # to m/s

    trelax_o2o = pom_bfm_parameters.nrt_o2o / pom_bfm_parameters.sec_per_day
    trelax_n1p = pom_bfm_parameters.nrt_n1p / pom_bfm_parameters.sec_per_day
    trelax_n3n = pom_bfm_parameters.nrt_n3n / pom_bfm_parameters.sec_per_day
    trelax_n4n = pom_bfm_parameters.nrt_n4n

    # LOOP OVER BFM STATE VAR'S
    for M in range(0,d3state.shape[1]):
    
        # ZEROING
        bfm_state_var = BfmStateVariables(pom_bfm_parameters.vertical_layers)

        # Load BFM state variable
        bfm_state_var.current[:-1] = d3state[:,M]
        bfm_state_var.backward[:-1] = d3stateb[:,M]

        bfm_state_var.current[-1] = bfm_state_var.current[-2]
        bfm_state_var.backward[-1] = bfm_state_var.backward[-2]

        sinking_velocity = W_ON*bfm_phys_vars.wgen + Weddy_ON*bfm_phys_vars.weddy
        
        # Surface and bottom fluxes for nutrients
        if 'O2o' in species and M == species['O2o']:      # Dissolved Oxygen (o2o)
            bfm_state_var.surface_flux = -(dOdt_wind[0] / constant_parameters.sec_per_day)
            bfm_state_var.bottom_flux = (d3state[-1,0] - nutrients.O2bott) * trelax_o2o
        elif 'O3c' in species and M == species['O3c']:    # Dissolved Inorganic Carbon (o3c)
            bfm_state_var.surface_flux = 0.
        elif 'N1p' in species and M == species['N1p']:    # Phosphate (n1p)
            bfm_state_var.surface_flux = 0.
            bfm_state_var.bottom_flux = (d3state[-1,1] - nutrients.PO4bott) * trelax_n1p
        elif 'N3n' in species and M == species['N3n']:    # Nitrate (n3n)
            bfm_state_var.surface_flux = 0.
            bfm_state_var.bottom_flux = (d3state[-1,2] - nutrients.NO3bott) * trelax_n3n
        elif 'N4n' in species and M == species['N4n']:    # Ammonium (n4n)
            bfm_state_var.surface_flux = 0.
            bfm_state_var.bottom_flux = nutrients.PONbott_grad*trelax_n4n
        elif 'N5s' in species and M == species['N5s']:    # Silicate (n5s)
            bfm_state_var.surface_flux = 0.
        else:
            bfm_state_var.surface_flux = 0.

        # Bottom flux for Dissolved Organic Matter (R1)
        # The botflux for Dissolved Organic Matter is left equal to ZERO

        # Bottom flux for Particulate Organic Matter  (R6)          
        # The botflux for Particulate Organic Matter is left equal to ZERO
            
        if include['r6']:
            if ('R6c' in species and M == species['R6c']) or ('R6n' in species and M == species['R6n']) \
                    or ('R6p' in species and M == species['R6p']) or ('R6s' in species and M == species['R6s']):
                detritus_sedimentation_rate = bfm_phys_vars.detritus_sedimentation
                sinking_velocity[:-1] -= detritus_sedimentation_rate/constant_parameters.sec_per_day

                # FINAL SINK VALUE
                sinking_velocity[-1] = sinking_velocity[-2]

        # Phytoplankton sedimentation        
        # The botflux for Phytoplankton is left equal to ZERO

        # FROM MODULEMEM --> iiP1 = 1, iiP2 = 2, 11P3 = 3, iiP4 = 4
        phyto_sedimentation_rates = bfm_phys_vars.phyto_sedimentation
        if ('P1c' in species and M == species['P1c']) or ('P1n' in species and M == species['P1n']) or ('P1p' in species and M == species['P1p']) \
                or ('P1l' in species and M == species['P1l']) or ('P1s' in species and M == species['P1s']):
            K = 1   # iiP1
            phyto_sedimentation_rates = bfm_phys_vars.phyto_sedimentation
            sinking_velocity[:-1] -=  phyto_sedimentation_rates[:,K-1]/constant_parameters.sec_per_day
            sinking_velocity[-1] = sinking_velocity[-2]
        elif ('P2c' in species and M == species['P2c']) or ('P2n' in species and M == species['P2n']) \
                or ('P2p' in species and M == species['P2p']) or ('P2l' in species and M == species['P2l']):
            K = 2   # iiP2
            phyto_sedimentation_rates = bfm_phys_vars.phyto_sedimentation
            sinking_velocity[:-1] -=  phyto_sedimentation_rates[:,K-1]/constant_parameters.sec_per_day
            sinking_velocity[-1] = sinking_velocity[-2]
        elif('P3c' in species and M == species['P3c']) or ('P3n' in species and M == species['P3n']) \
                or ('P3p' in species and M == species['P3p']) or ('P3l' in species and M == species['P3l']):
            K = 3   # iiP3
            phyto_sedimentation_rates = bfm_phys_vars.phyto_sedimentation
            sinking_velocity[:-1] -=  phyto_sedimentation_rates[:,K-1]/constant_parameters.sec_per_day
            sinking_velocity[-1] = sinking_velocity[-2]
        elif ('P4c' in species and M == species['P4c']) or ('P4n' in species and M == species['P4n']) \
                or ('P4p' in species and M == species['P4p']) or ('P4l' in species and M == species['P4l']):
            K = 4   # iiP4
            phyto_sedimentation_rates = bfm_phys_vars.phyto_sedimentation
            sinking_velocity[:-1] -=  phyto_sedimentation_rates[:,K-1]/constant_parameters.sec_per_day
            sinking_velocity[-1] = sinking_velocity[-2]
        
        # Bottom flux for Zooplankton
        # The botflux for Zooplankton is left equal to ZERO

        # Sinking: upstream vertical advection
        bfm_state_var = vertical_advection(bfm_state_var,sinking_velocity,vertical_grid)

        # Source splitting (SoS) leapfrog integration
        for i in range(0,pom_bfm_parameters.vertical_layers-1):
            bfm_state_var.forward[i] = bfm_state_var.backward[i] + pom_bfm_parameters.dti2*((bfm_state_var.forward[i]/pom_bfm_parameters.h) + bfm_rates[i,M])
        
        # Compute vertical diffusion and terminate integration (implicit leapfrogging)
        bfm_state_var = temperature_and_salinity_profiles(pom_bfm_parameters, vertical_grid, diffusion, bfm_state_var, 0, 'BFM')
        
        # Clipping (if needed)
        for i in range(0,pom_bfm_parameters.num_boxes):
            bfm_state_var.forward[i] = max(constant_parameters.p_small,bfm_state_var.forward[i])

        # Mix the time step and restore time sequence
        d3stateb[:,M] = bfm_state_var.current[:-1] + 0.5*pom_bfm_parameters.smoth*(bfm_state_var.forward[:-1] + bfm_state_var.backward[:-1] - 2.*bfm_state_var.current[:-1])
        d3state[:,M] = bfm_state_var.forward[:-1]
    
    if not bfm_parameters.AssignAirPelFluxesInBFMFlag:
        dOdt_wind[:] = 0.
        do3cdt_air_sea_flux[:] = 0.

    return d3state, d3stateb


def vertical_extinction(bfm_phys_vars, d3state, species): #, group):
    """
    Description: Computes the z-dependent light vertical extinction coefficients
    """
    # Calcluations taken from bfm/bfm50/Functions/phyto.py - Updated from 0D to 1D
    if 'P1l' in species:    p1l = d3state[:,species['P1l']]
    else:   p1l = np.zeros(pom_bfm_parameters.num_boxes)

    if 'P2l' in species:    p2l = d3state[:,species['P2l']]
    else:   p2l = np.zeros(pom_bfm_parameters.num_boxes)

    if 'P3l' in species:    p3l = d3state[:,species['P3l']]
    else:   p3l = np.zeros(pom_bfm_parameters.num_boxes)
    
    if 'P4l' in species:    p4l = d3state[:,species['P4l']]
    else:   p4l = np.zeros(pom_bfm_parameters.num_boxes)

    if 'R6c' in species:    r6c = d3state[:,species['R6c']]
    else:   r6c = np.zeros(pom_bfm_parameters.num_boxes)
    
    vertical_extinction = (env_parameters.p_eps0 + env_parameters.p_epsESS*bfm_phys_vars.suspended_matter + env_parameters.p_epsR6*r6c) + (phyto1_parameters.c_P * p1l) + (phyto2_parameters.c_P * p2l) + (phyto3_parameters.c_P * p3l) + (phyto4_parameters.c_P * p4l)

    return vertical_extinction
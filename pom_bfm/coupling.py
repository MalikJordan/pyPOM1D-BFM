import numpy as np
from bfm.bfm50.bfm50_rate_eqns import bfm50_rate_eqns
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


def average_fields(d3ave, d3state, chl, npp, case):
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
    oxy = d3state[:,0]
    nit = d3state[:,2]
    phos = d3state[:,1]
    pon = d3state[:,11] + d3state[:,16] + d3state[:,20] + d3state[:,24] + d3state[:,28] + d3state[:,31] + d3state[:,34] + d3state[:,37] + d3state[:,40] + d3state[:,45]
    dic = d3state[:,48]

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
    # p_burvel_PI   [m/d]   Botttom burial velocity for plankton
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


def pom_bfm_1d(i, vertical_grid, time, diffusion, nutrients, bfm_phys_vars, d3state, d3stateb, d3ave):
    """
    Description: Handles the calling of all the subroutines that provides BFM with all the needed information about the physical environment
    """
    bfm_rates = np.zeros((pom_bfm_parameters.num_boxes,bfm_parameters.num_d3_box_states))

    dOdt_wind = np.zeros(pom_bfm_parameters.num_boxes)
    do3cdt_air_sea_flux = np.zeros(pom_bfm_parameters.num_boxes)
    
    bfm_rates, bfm_phys_vars, dOdt_wind, do3cdt_air_sea_flux, chl, npp = bfm50_rate_eqns(i, time, d3state, bfm_phys_vars, seasonal_cycle=False)

    d3state, d3stateb = vertical_diffusivity(vertical_grid, diffusion, nutrients, d3state, d3stateb, bfm_rates, bfm_phys_vars, dOdt_wind, do3cdt_air_sea_flux)
    
    d3ave.count += 1    # count initialized to 0, update before averaging
    if (d3ave.count > 0) and (d3ave.count < (pom_bfm_parameters.sec_per_day/pom_bfm_parameters.dti)):
        d3ave = average_fields(d3ave, d3state, chl, npp, 'Accumulate')
    elif d3ave.count == (pom_bfm_parameters.sec_per_day/pom_bfm_parameters.dti):
        d3ave = average_fields(d3ave, d3state, chl, npp, 'Mean')

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


def vertical_diffusivity(vertical_grid, diffusion, nutrients, d3state, d3stateb, bfm_rates, bfm_phys_vars, dOdt_wind, do3cdt_air_sea_flux):
    """
    Description: Calculates the vertical diffusivity of BFM biochemical components and
                 integrats BFM state variables with Source Splitting (SoS) method
    """
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   GLOBAL DEFINITION OF PELAGIC (D3/D2) STATE VARIABLES (From ModuleMem)
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Values correspond to index from bfm.variable_info.py
    ppo2o = 0; ppn1p = 1; ppn3n = 2
    ppn4n = 3; ppo4n = 4; ppn5s = 5; ppn6r = 6; ppb1c = 7; ppb1n = 8; ppb1p = 9; ppp1c = 10
    ppp1n = 11; ppp1p = 12; ppp1l = 13; ppp1s = 14; ppp2c = 15; ppp2n = 16; ppp2p = 17
    ppp2l = 18; ppP2s = 0; ppp3c = 19; ppp3n = 20; ppp3p = 21; ppp3l = 22; ppP3s = 0
    ppp4c = 23; ppp4n = 24; ppp4p = 25; ppp4l = 26; ppP4s = 0; ppz3c = 27; ppz3n = 28
    ppz3p = 29; ppz4c = 30; ppz4n = 31; ppz4p = 32; ppz5c = 33; ppz5n = 34; ppz5p = 35
    ppz6c = 36; ppz6n = 37; ppz6p = 38; ppr1c = 39; ppr1n = 40; ppr1p = 41; ppR1s = 0
    ppr2c = 42; ppR2n = 0; ppR2p = 0; ppR2s = 0; ppr3c = 43; ppR3n = 0; ppR3p = 0; ppR3s = 0
    ppr6c = 44; ppr6n = 45; ppr6p = 46; ppr6s = 47; ppo3c = 48; ppo3h = 49

    # Local variables
    # BFM state variable @ time t-dti, t, t+dti respectively
    # bfm_state_var = BfmStateVariables()

    # Sedumentation velocity
    # sinking_velocity = np.zeros(pom_bfm_parameters.vertical_layers)

    # The input general cir. vertical vel. is suppose to be in m/s
    W_ON = 1.0

    # The input eddy vertical vel. is provided in m/d
    Weddy_ON = 0.1/86400.0  # to m/s

    trelax_o2o = pom_bfm_parameters.nrt_o2o / pom_bfm_parameters.sec_per_day
    trelax_n1p = pom_bfm_parameters.nrt_n1p / pom_bfm_parameters.sec_per_day
    trelax_n3n = pom_bfm_parameters.nrt_n3n / pom_bfm_parameters.sec_per_day
    trelax_n4n = pom_bfm_parameters.nrt_n4n

    # LOOP OVER BFM STATE VAR'S
    for M in range(0,bfm_parameters.num_d3_box_states):
    
        # ZEROING
        bfm_state_var = BfmStateVariables(pom_bfm_parameters.vertical_layers)
        POCsink = 0.

        # Load BFM state variable
        bfm_state_var.current[:-1] = d3state[:,M]
        bfm_state_var.backward[:-1] = d3stateb[:,M]

        bfm_state_var.current[-1] = bfm_state_var.current[-2]
        bfm_state_var.backward[-1] = bfm_state_var.backward[-2]

        sinking_velocity = W_ON*bfm_phys_vars.wgen + Weddy_ON*bfm_phys_vars.weddy
        
        # Surface and bottom fluxes for nutrients
        if M == ppo2o:      # Dissolved Oxygen (o2o)
            bfm_state_var.surface_flux = -(dOdt_wind[0] / constant_parameters.sec_per_day)
            bfm_state_var.bottom_flux = (d3state[-1,0] - nutrients.O2bott) * trelax_o2o
        elif M == ppo3c:    # Dissolved Inorganic Carbon (o3c)
            bfm_state_var.surface_flux = 0.
        elif M == ppn1p:    # Phosphate (n1p)
            bfm_state_var.surface_flux = 0.
            bfm_state_var.bottom_flux = (d3state[-1,1] - nutrients.PO4bott) * trelax_n1p
        elif M == ppn3n:    # Nitrate (n3n)
            bfm_state_var.surface_flux = 0.
            bfm_state_var.bottom_flux = (d3state[-1,2] - nutrients.NO3bott) * trelax_n3n
        elif M == ppn4n:    # Ammonium (n4n)
            bfm_state_var.surface_flux = 0.
            bfm_state_var.bottom_flux = nutrients.PONbott_grad*trelax_n4n
        elif M == ppn5s:    # Silicate (n5s)
            bfm_state_var.surface_flux = 0.
        else:
            bfm_state_var.surface_flux = 0.

        # Bottom flux for Dissolved Organic Matter (R1)
        # The botflux for Dissolved Organic Matter is left equal to ZERO

        # Bottom flux for Particulate Organic Matter  (R6)          
        # The botflux for Particulate Organic Matter is left equal to ZERO
        if ppr6c <= M <= ppr6s:
            detritus_sedimentation_rate = bfm_phys_vars.detritus_sedimentation
            sinking_velocity[:-1] -= detritus_sedimentation_rate/constant_parameters.sec_per_day

            # FINAL SINK VALUE
            sinking_velocity[-1] = sinking_velocity[-2]

        # Phytoplankton sedimentation        
        # The botflux for Phytoplankton is left equal to ZERO
        if ppp1c <= M <= ppp4l:
            # phyto_sedimentation_rates = phyto_sedimentation()
            # FROM MODULEMEM --> iiP1 = 1, iiP2 = 2, 11P3 = 3, iiP4 = 4
            phyto_sedimentation_rates = bfm_phys_vars.phyto_sedimentation
            if M in range(ppp1c,ppp1s):
                K = 1   # iiP1
            elif M in range(ppp2c,ppp2l):
                K = 2   # iiP2
            elif M in range(ppp3c,ppp3l):
                K = 3   # iiP3
            elif M in range(ppp4c,ppp4l):
                K = 4   # iiP4

            sinking_velocity[:-1] -=  phyto_sedimentation_rates[:,K-1]/constant_parameters.sec_per_day

            # FINAL SINK VALUE
            sinking_velocity[-1] = sinking_velocity[-2]

        # Bottom flux for Zooplankton
        # The botflux for Zooplankton is left equal to ZERO

        # Sinking: upstream vertical advection
        bfm_state_var = vertical_advection(bfm_state_var,sinking_velocity,vertical_grid)

        # Source splitting (SoS) leapfrog integration
        for i in range(0,pom_bfm_parameters.vertical_layers-1):
            bfm_state_var.forward[i] = bfm_state_var.backward[i] + pom_bfm_parameters.dti2*((bfm_state_var.forward[i]/pom_bfm_parameters.h) + bfm_rates[M,i])
        
        # Compute vertical diffusion and terminate integration (implicit leapfrogging)
        bfm_state_var = temperature_and_salinity_profiles(pom_bfm_parameters, vertical_grid, diffusion, bfm_state_var, 0, 'BFM')
        
        # Clipping (if needed)
        for i in range(0,pom_bfm_parameters.num_boxes):
            bfm_state_var.forward[i] = max(constant_parameters.p_small,bfm_state_var.forward[i])

        d3stateb[:,M] = bfm_state_var.current[:-1] + 0.5*pom_bfm_parameters.smoth*(bfm_state_var.forward[:-1] + bfm_state_var.backward[:-1] - 2.*bfm_state_var.current[:-1])
        d3state[:,M] = bfm_state_var.forward[:-1]
    
    if not bfm_parameters.AssignAirPelFluxesInBFMFlag:
        dOdt_wind[:] = 0.
        do3cdt_air_sea_flux[:] = 0.

    return d3state, d3stateb


def vertical_extinction(bfm_phys_vars, d3state):
    """
    Description: Computes the z-dependent light vertical extinction coefficients
    """
    # Calcluations taken from bfm/bfm50/Functions/phyto.py - Updated from 0D to 1D

    vertical_extinction = (env_parameters.p_eps0 + env_parameters.p_epsESS*bfm_phys_vars.suspended_matter + env_parameters.p_epsR6*d3state[:,44]) + (phyto1_parameters.c_P * d3state[:,13]) + (phyto2_parameters.c_P * d3state[:,18]) + (phyto3_parameters.c_P * d3state[:,22]) + (phyto4_parameters.c_P * d3state[:,26])

    return vertical_extinction
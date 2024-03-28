import numpy as np
import os
from pom.data_classes import MonthlyForcing, Nutrients, Stresses
from pom.initialization import read_pom_input
from pom.parameters import PomBfm


def forcing_manager(time_loop_counter,counters,month1_data,month2_data,pom_bfm_parameters):
    """
    Description: Handles the forcing data reading and interpolation.
                 Accomodates perpetual year monthly forcing time series.
                 Data currently handled are time series of monthly averaged data.
                 
    :return: forcing data
    """
    # INITIALISATION AND FIRST FORCING READING
    if time_loop_counter == 0:

        # Month and day counters
        counters.day_counter = 1
        counters.month_counter = 0

        # Timesteps to cover one day/month
        counters.timesteps_per_day = pom_bfm_parameters.sec_per_day / pom_bfm_parameters.dti
        counters.timesteps_per_month = 30 * counters.timesteps_per_day

        # Month and day interpolators
        counters.day_interpolator = -1
        counters.month_interpolator = (counters.timesteps_per_month / 2.) - 1    # Climatological forcing data centered at day 15 of each month

        # Reading for the first month
        month1_data = write_forcing_data(counters.month_counter)

        # Update the day counter
        counters.day_counter = counters.day_counter + 1

        # Update the month counter
        counters.month_counter = counters.month_counter + 1

        # Reading for the second month
        month2_data = write_forcing_data(counters.month_counter)

    # Update interpolation counters
    counters.day_interpolator = counters.day_interpolator + 1
    counters.ratio_day = counters.day_interpolator / counters.timesteps_per_day

    counters.month_interpolator = counters.month_interpolator + 1
    counters.ratio_month = counters.month_interpolator / counters.timesteps_per_month

    # Interpolate wind stress
    wind_stress = Stresses()
    wind_stress.zonal = month1_data.wsu + counters.ratio_month * (month2_data.wsu - month1_data.wsu)
    wind_stress.meridional = month1_data.wsv + counters.ratio_month * (month2_data.wsv - month1_data.wsv)

    # Interpolate heat flux
    if pom_bfm_parameters.idiagn == 0:
        wtsurf = month1_data.wtsurf + counters.ratio_month * (month2_data.wtsurf - month1_data.wtsurf)
        swrad = month1_data.swrad + counters.ratio_month * (month2_data.swrad - month1_data.swrad)
    else:
        wtsurf = 0  # not needed for diagnostic mode (idiagn = 1), see 4.6.5 in manual
        swrad = month1_data.swrad + counters.ratio_month * (month2_data.swrad - month1_data.swrad)



    # Interpolate temperature and salinity profiles
    tstar = month1_data.tclim + counters.ratio_month * (month2_data.tclim - month1_data.tclim)
    sstar = month1_data.sclim + counters.ratio_month * (month2_data.sclim - month1_data.sclim)
    wgen  = month1_data.wclim + counters.ratio_month * (month2_data.wclim - month1_data.wclim)

    if counters.ratio_month <= 0.5:
        weddy = month1_data.weddy1
    else:
        weddy = month1_data.weddy2

    if pom_bfm_parameters.idiagn == 0:
        tsurf = tstar[0]
        ssurf = sstar[0]
    elif pom_bfm_parameters.idiagn == 1:
        tf = tstar
        sf = sstar

    # Interpolate suspended inorganic matter
    ism = month1_data.ism + counters.ratio_month * (month2_data.ism - month1_data.ism)

    # Interpolate surface and bottom nutrients
    nutrients = Nutrients()

    nutrients.NO3surf = month1_data.NO3_s + counters.ratio_month * (month2_data.NO3_s - month1_data.NO3_s)
    nutrients.NH4surf = month1_data.NH4_s + counters.ratio_month * (month2_data.NH4_s - month1_data.NH4_s)
    nutrients.PO4surf = month1_data.PO4_s + counters.ratio_month * (month2_data.PO4_s - month1_data.PO4_s)
    nutrients.SIO4surf = month1_data.SIO4_s + counters.ratio_month * (month2_data.SIO4_s - month1_data.SIO4_s)

    nutrients.O2bott = month1_data.O2_b + counters.ratio_month * (month2_data.O2_b - month1_data.O2_b)
    nutrients.NO3bott = month1_data.NO3_b + counters.ratio_month * (month2_data.NO3_b - month1_data.NO3_b)
    nutrients.PO4bott = month1_data.PO4_b + counters.ratio_month * (month2_data.PO4_b - month1_data.PO4_b)
    nutrients.PONbott_grad = month1_data.PON_b + counters.ratio_month * (month2_data.PON_b - month1_data.PON_b)

    if counters.month_interpolator == counters.timesteps_per_month:

        # Update the month counter
        if os.path.exists('output_file.txt'):
            output_file = open('output_file.txt','a')
            output_file.write('month_counter = ')
            output_file.write(str(counters.month_counter))
            output_file.write('\n')
            output_file.close()
        else:
            output_file = open('output_file.txt','w')
            output_file.write('month_counter = ')
            output_file.write(str(counters.month_counter))
            output_file.write('\n')
            output_file.close()

        print('month_counter = ',counters.month_counter)
        counters.month_counter = counters.month_counter + 1
        
        # Reset the interpolator
        counters.month_interpolator = 0

        # Shift the monthly data
        month1_data = month2_data

        # If 12 months have passed, restart the reading sequence
        if counters.month_counter > 12:
            counters.month_counter = 0
            month1_data = write_forcing_data(counters.month_counter)

            counters.month_counter = counters.month_counter + 1

        # Read the following month
        month2_data = write_forcing_data(counters.month_counter)

    return tf, tstar, sf, sstar, swrad, wtsurf, wind_stress, wgen, weddy, month1_data, month2_data, counters, nutrients, ism



def write_forcing_data(month_counter):
    """
    Definition: Writes forcing data for each month.
    """
    pom_bfm_parameters = PomBfm()
    month_data = MonthlyForcing(pom_bfm_parameters.vertical_layers)

    wind_speed_zonal, wind_speed_meridional, surface_salinity, solar_radiation, inorganic_suspended_matter, \
        salinity_climatology, temperature_climatology, w_velocity_climatology, w_eddy_velocity_1, \
        w_eddy_velocity_2, salinity, temperature, shortwave_radiation, surface_heat_flux, kinetic_energy_loss, \
        NO3_surf, NH4_surf, PO4_surf, SIO4_surf, O2_bott, NO3_bott, PO4_bott, PON_bott                  = read_pom_input(pom_bfm_parameters.vertical_layers)
    
    # Climatology
    month_data.sclim = salinity_climatology[:,month_counter]
    month_data.tclim = temperature_climatology[:,month_counter]
    month_data.wclim = w_velocity_climatology[:,month_counter]

    # Intermittant eddy w velocities
    month_data.weddy1 = w_eddy_velocity_1[:,month_counter]
    month_data.weddy2 = w_eddy_velocity_2[:,month_counter]

    # Inorganic suspended matter
    month_data.ism = inorganic_suspended_matter[:-1,month_counter]

    month_data.wsu = -0.001 * wind_speed_zonal[month_counter]          # N/m2-->m2/s2
    month_data.wsv = -0.001 * wind_speed_meridional[month_counter]     # N/m2-->m2/s2
    
    # Heat Flux
    month_data.swrad  = -shortwave_radiation[month_counter] / pom_bfm_parameters.water_specific_heat_times_density     # W/m2-->deg.C*m/s
    month_data.wtsurf = -surface_heat_flux[month_counter] / pom_bfm_parameters.water_specific_heat_times_density       # W/m2-->deg.C*m/s
    month_data.qcorr  = kinetic_energy_loss[month_counter]

    # Surface nutrients
    month_data.NO3_s  = NO3_surf[month_counter]
    month_data.NH4_s  = NH4_surf[month_counter]
    month_data.PO4_s  = PO4_surf[month_counter]
    month_data.SIO4_s = SIO4_surf[month_counter]

    # Bottom nutrients
    month_data.O2_b  = O2_bott[month_counter]
    month_data.NO3_b = NO3_bott[month_counter]
    month_data.PO4_b = PO4_bott[month_counter]
    month_data.PON_b = PON_bott[month_counter]

    return month_data
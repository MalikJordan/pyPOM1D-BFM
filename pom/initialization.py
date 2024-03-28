import numpy as np
from os import path
from pom.parameters import PomInputFiles

def read_pom_input(vertical_layers):
    """
    Description: Opens forcing files reading the paths specified in the pom_input namelist.

    :return: data arrays for wind stress, surface salinity, solar radiation, inorganic
             suspended matter, salinity and temperature vertical profiles, general circulation
             for w velocity, intermediate eddy velocities, salinity and temperature initial
             conditions, heat flux loss, and surface and bottom nutrients
    """

    input_files = PomInputFiles()
    
    # Length of input arrays
    array_length = 13   # months (D-J-F-M-A-M-J-J-A-S-O-N-D)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   Wind speed (u,v)
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if path.exists(input_files.wind_stress):
        wind_speed_data = np.fromfile(input_files.wind_stress)
    wind_speed_zonal   = np.zeros(array_length)
    wind_speed_meridional   = np.zeros(array_length)
    for i in range(0,array_length):
        wind_speed_zonal[i] = wind_speed_data[2*i + 0]
        wind_speed_meridional[i] = wind_speed_data[2*i + 1]

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   Surface salinity
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if path.exists(input_files.surface_salinity):
        surface_salinity = np.fromfile(input_files.surface_salinity)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   Radiance
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if path.exists(input_files.shortwave_solar_radiation):
        solar_radiation = np.fromfile(input_files.shortwave_solar_radiation)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   Inorganic suspended matter
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if path.exists(input_files.inorganic_suspended_matter):
        inorganic_suspended_matter_data = np.fromfile(input_files.inorganic_suspended_matter)
    inorganic_suspended_matter   = np.zeros((vertical_layers,array_length))
    for i in range(0,array_length):
        for x in range(0, vertical_layers):
            inorganic_suspended_matter[x,i] = inorganic_suspended_matter_data[vertical_layers * i + x]

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   Salinity climatology (DIAGNOSTIC MODE)
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if path.exists(input_files.salinity_vertical_profile):
        salinity_vertical_profile_data = np.fromfile(input_files.salinity_vertical_profile)
    salinity_climatology = np.zeros((vertical_layers,array_length))
    for i in range(0,array_length):
        for x in range(0, vertical_layers):
            salinity_climatology[x,i] = salinity_vertical_profile_data[vertical_layers * i + x]

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   Temperature climatology (DIAGNOSTIC MODE)
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if path.exists(input_files.temperature_vertical_profile):
        temperature_vertical_profile_data = np.fromfile(input_files.temperature_vertical_profile)
    temperature_climatology = np.zeros((vertical_layers,array_length))
    for i in range(0,array_length):
        for x in range(0, vertical_layers):
            temperature_climatology[x,i] = temperature_vertical_profile_data[vertical_layers * i + x]

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   General circulation w velocity climatology
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if path.exists(input_files.general_circulation_w_velocity):
        general_circulation_w_velocity_data = np.fromfile(input_files.general_circulation_w_velocity)
    w_velocity_climatology  = np.zeros((vertical_layers,array_length))
    for i in range(0,array_length):
        for x in range(0, vertical_layers):
            w_velocity_climatology[x,i] = general_circulation_w_velocity_data[vertical_layers * i + x]

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   Intermittant eddy w velocity 1
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if path.exists(input_files.intermediate_eddy_w_velocity_1):
        intermediate_eddy_w_velocity_1_data = np.fromfile(input_files.intermediate_eddy_w_velocity_1)
    w_eddy_velocity_1  = np.zeros((vertical_layers,array_length))
    for i in range(0,array_length):
        for x in range(0, vertical_layers):
            w_eddy_velocity_1[x,i] = intermediate_eddy_w_velocity_1_data[vertical_layers * i + x]

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   Intermittant eddy w velocity 2
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if path.exists(input_files.intermediate_eddy_w_velocity_2):
        intermediate_eddy_w_velocity_2_data = np.fromfile(input_files.intermediate_eddy_w_velocity_2)
    w_eddy_velocity_2 = np.zeros((vertical_layers,array_length))
    for i in range(0,array_length):
        for x in range(0, vertical_layers):
            w_eddy_velocity_2[x,i] = intermediate_eddy_w_velocity_2_data[vertical_layers * i + x]

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   Salinity initial profile
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if path.exists(input_files.salinity_initial_conditions):
        salinity_backward = np.fromfile(input_files.salinity_initial_conditions)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   Temperature initial profile
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if path.exists(input_files.temperature_initial_conditions):
        temperature_backward = np.fromfile(input_files.temperature_initial_conditions)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   Heat flux
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if path.exists(input_files.heat_flux_loss):
        heat_flux_loss_data = np.fromfile(input_files.heat_flux_loss)
    shortwave_radiation = np.zeros(array_length)
    surface_heat_flux = np.zeros(array_length)
    kinetic_energy_loss = np.zeros(array_length)
    for i in range(0,array_length):
        shortwave_radiation[i]  = heat_flux_loss_data[3*i + 0]
        surface_heat_flux[i] = heat_flux_loss_data[3*i + 1]
        kinetic_energy_loss[i]  = heat_flux_loss_data[3*i + 2]

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   Surface nutrients
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if path.exists(input_files.surface_nutrients):
        surface_nutrients_data  = np.fromfile(input_files.surface_nutrients)
    NO3_s1  = np.zeros(array_length)
    NH4_s1  = np.zeros(array_length)
    PO4_s1  = np.zeros(array_length)
    SIO4_s1 = np.zeros(array_length)
    for i in range(0,array_length):
        NO3_s1[i]  = surface_nutrients_data[4*i + 0]
        NH4_s1[i]  = surface_nutrients_data[4*i + 1]
        PO4_s1[i]  = surface_nutrients_data[4*i + 2]
        SIO4_s1[i] = surface_nutrients_data[4*i + 3]

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   Bottom nutrients
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if path.exists(input_files.bottom_nutrients):
        bottom_nutrients_data = np.fromfile(input_files.bottom_nutrients)
    O2_b1   = np.zeros(array_length)
    NO3_b1  = np.zeros(array_length)
    PO4_b1  = np.zeros(array_length)
    PON_b1  = np.zeros(array_length)
    for i in range(0,array_length):
        O2_b1[i]  = bottom_nutrients_data[4*i + 0]
        NO3_b1[i] = bottom_nutrients_data[4*i + 1]
        PO4_b1[i] = bottom_nutrients_data[4*i + 2]
        PON_b1[i] = bottom_nutrients_data[4*i + 3]

    return wind_speed_zonal, wind_speed_meridional, surface_salinity, solar_radiation, inorganic_suspended_matter, \
           salinity_climatology, temperature_climatology, w_velocity_climatology, w_eddy_velocity_1, \
           w_eddy_velocity_2, salinity_backward, temperature_backward, \
           shortwave_radiation, surface_heat_flux, kinetic_energy_loss, \
           NO3_s1, NH4_s1, PO4_s1, SIO4_s1, O2_b1, NO3_b1, PO4_b1, PON_b1    


def read_temperature_and_salinity_initial_coditions(vertical_layers):
    """
    Description: Opens and reads files containing the initial conditions for temperature
                 and salinity. Files are read from the pom_input namelist.

    :return: temperature and salinity at the current and backward time levels
    """
    input_files = PomInputFiles()
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   SALINITY INITIAL PROFILE
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if path.exists(input_files.salinity_initial_conditions):
        salinity_backward = np.fromfile(input_files.salinity_initial_conditions)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   TEMPERATURE INITIAL PROFILE
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if path.exists(input_files.temperature_initial_conditions):
        temperature_backward = np.fromfile(input_files.temperature_initial_conditions)

    temperature = np.zeros(vertical_layers)
    salinity = np.zeros(vertical_layers)

    temperature[:] = temperature_backward[:]
    salinity[:] = salinity_backward[:]

    return temperature, temperature_backward, \
           salinity, salinity_backward

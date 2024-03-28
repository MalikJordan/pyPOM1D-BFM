import numpy as np


def density_profile(pom_bfm_parameters,temperature,salinity,vertical_grid):

    gravity = 9.806
    vertical_density_profile = np.zeros(pom_bfm_parameters.vertical_layers)

    pressure = -gravity * 1.025 * vertical_grid.vertical_spacing_staggered[:-1] * pom_bfm_parameters.h * 0.01

    cr = 1449.1 + (0.0821*pressure) + (4.55*temperature.backward[:-1]) - (0.045*np.power(temperature.backward[:-1],2)) + (1.34*(salinity.backward[:-1] - 35.0))
    cr = pressure/np.power(cr,2)
    
    density = 999.842594 + (6.793952e-02*temperature.backward[:-1]) - (9.095290e-03*np.power(temperature.backward[:-1],2)) \
            + (1.001685e-04*np.power(temperature.backward[:-1],3)) - (1.120083e-06*np.power(temperature.backward[:-1],4)) + (6.536332e-09*np.power(temperature.backward[:-1],5)) \
            + (0.824493 - (4.0899e-03*temperature.backward[:-1]) + (7.6438e-05*np.power(temperature.backward[:-1],2))
                    - (8.2467e-07*np.power(temperature.backward[:-1],3)) + (5.3875e-09*np.power(temperature.backward[:-1],4))) * salinity.backward[:-1] \
            + (-5.72466e-03 + 1.0227e-04*temperature.backward[:-1] - (1.6546e-06*np.power(temperature.backward[:-1],2))) * (np.power(np.abs(salinity.backward[:-1]),1.5)) \
                    + (4.8314e-04*np.power(salinity.backward[:-1],2)) + 1.0e05*cr*(1.0 - (2*cr))
    
    vertical_density_profile[:-1] = (density - 1000.) * 1.e-03
    vertical_density_profile[-1] = vertical_density_profile[-2]

    return vertical_density_profile


def kinetic_energy_profile(pom_bfm_parameters,vertical_grid, diffusion, density, zonal_velocity, meridional_velocity, 
                           kinetic_energy, kinetic_energy_times_length, wind_stress, bottom_stress):

    # INITIALIZE VARIABLES
    A = np.zeros(pom_bfm_parameters.vertical_layers)
    C = np.zeros(pom_bfm_parameters.vertical_layers)
    VH = np.zeros(pom_bfm_parameters.vertical_layers)
    VHP = np.zeros(pom_bfm_parameters.vertical_layers)

    KN = np.zeros(pom_bfm_parameters.vertical_layers)
    GH = np.zeros(pom_bfm_parameters.vertical_layers)
    SH = np.zeros(pom_bfm_parameters.vertical_layers)
    SM = np.zeros(pom_bfm_parameters.vertical_layers)

    DTEF = np.zeros(pom_bfm_parameters.vertical_layers)
    BPROD = np.zeros(pom_bfm_parameters.vertical_layers)
    PROD = np.zeros(pom_bfm_parameters.vertical_layers)
    SPROD = np.zeros(pom_bfm_parameters.vertical_layers)
    pressure = 0.

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   LOCAL ARRAYS
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    BOYGR = np.zeros(pom_bfm_parameters.vertical_layers); CC = np.zeros(pom_bfm_parameters.vertical_layers)
    TEMP1 = np.zeros(pom_bfm_parameters.vertical_layers); TEMP2 = np.zeros(pom_bfm_parameters.vertical_layers); TEMP3 = np.zeros(pom_bfm_parameters.vertical_layers)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   DATA STATEMENTS
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    A1 = 0.92
    B1 = 16.6
    A2 = 0.74
    B2 = 10.1
    C1 = 0.08
    E1 = 1.8
    E2 = 1.33
    E3 = 1.0
    von_karman_constant = 0.40  # KAPPA
    SQ = 0.2
    CIWC = 1.0
    gravity = 9.806
    SMALL = 1.E-08
    
    A[1:-1] = -pom_bfm_parameters.dti2 * (diffusion.kinetic_energy[2:] + diffusion.kinetic_energy[1:-1] + 2 * pom_bfm_parameters.umol) * \
                0.5 / (vertical_grid.vertical_spacing_staggered[:-2] * vertical_grid.vertical_spacing[1:-1] * pom_bfm_parameters.h * pom_bfm_parameters.h)
    C[1:-1] = -pom_bfm_parameters.dti2 * (diffusion.kinetic_energy[:-2] + diffusion.kinetic_energy[1:-1] + 2 * pom_bfm_parameters.umol) * \
                0.5 / (vertical_grid.vertical_spacing_staggered[:-2] * vertical_grid.vertical_spacing[:-2] * pom_bfm_parameters.h * pom_bfm_parameters.h)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   THE FOLLOWING SECTION SOLVES FOR THE EQUATION
    #   DT2*(KQ*Q2')' - Q2*(2.*DT2*DTEF+1.) = -Q2B
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    CONST1 = 16.6 ** 0.6666667 * CIWC

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   BOUNDARY CONDITIONS
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    VH[0] = 0.0
    VHP[0] = np.sqrt(wind_stress.zonal ** 2 + wind_stress.meridional ** 2) * CONST1
    kinetic_energy.forward[pom_bfm_parameters.vertical_layers - 1] = 0.5 * np.sqrt((bottom_stress.zonal + bottom_stress.zonal) ** 2 +
                                                                (bottom_stress.meridional + bottom_stress.meridional) ** 2) * CONST1

    
    kinetic_energy.backward[1:-1] = np.abs(kinetic_energy.backward[1:-1])
    kinetic_energy_times_length.backward[1:-1] = np.abs(kinetic_energy_times_length.backward[1:-1])
    BOYGR[1:-1] = gravity * (density[:-2] - density[1:-1]) / (vertical_grid.vertical_spacing_staggered[:-2] * pom_bfm_parameters.h)
    DTEF[1:-1] = kinetic_energy.backward[1:-1] * np.sqrt(kinetic_energy.backward[1:-1]) / (B1 * kinetic_energy_times_length.backward[1:-1] + SMALL)
    SPROD[1:-1] = .25 * diffusion.momentum[1:-1] * \
                   ((zonal_velocity.current[1:-1] + zonal_velocity.current[1:-1] - zonal_velocity.current[0:-2] - zonal_velocity.current[0:-2]) ** 2
                    + (meridional_velocity.current[1:-1] + meridional_velocity.current[1:-1] - meridional_velocity.current[0:-2] - meridional_velocity.current[0:-2]) ** 2) / \
                   (vertical_grid.vertical_spacing_staggered[0:-2] * pom_bfm_parameters.h) ** 2 * CIWC ** 2
    BPROD[1:-1] = diffusion.tracers[1:-1] * BOYGR[1:-1]
    PROD[1:-1] = SPROD[1:-1] + BPROD[1:-1]
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   SWEEP DOWNWARD
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    for i in range(1, pom_bfm_parameters.vertical_layers - 1):
        VHP[i] = 1. / (A[i] + C[i] * (1. - VH[i - 1]) - (2. * pom_bfm_parameters.dti2 * DTEF[i] + 1.))
        VH[i] = A[i] * VHP[i]
        VHP[i] = (-2. * pom_bfm_parameters.dti2 * PROD[i] + C[i] * VHP[i - 1] - kinetic_energy.backward[i]) * VHP[i]
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   SWEEP UPWARD
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    for i in range(2, pom_bfm_parameters.vertical_layers+1):  # 104
        k = pom_bfm_parameters.vertical_layers - i
        kinetic_energy.forward[k] = VH[k] * kinetic_energy.forward[k + 1] + VHP[k]

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   THE FOLLOWING SEECTION SOLVES FOR TEH EQUATION
    #   DT2(KQ*Q2L')' - Q2L*(DT2*DTEF+1.) = -Q2LB
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   BOUNDARY CONDITIONS
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    VH[0] = 0.
    VHP[0] = 0.
    kinetic_energy_times_length.forward[pom_bfm_parameters.vertical_layers - 1] = 0.

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   SWEEP DOWNWARD
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    for i in range(1, pom_bfm_parameters.vertical_layers - 1):
        DTEF[i] = DTEF[i] * (1. + E2 * ((1. / np.abs(vertical_grid.vertical_coordinates[i] - vertical_grid.vertical_coordinates[0])
                                         + 1. / np.abs(vertical_grid.vertical_coordinates[i] - vertical_grid.vertical_coordinates[pom_bfm_parameters.vertical_layers-1]))
                                        * vertical_grid.length_scale[i] / (pom_bfm_parameters.h * von_karman_constant)) ** 2)
        VHP[i] = 1. / (A[i] + C[i] * (1. - VH[i - 1]) - (pom_bfm_parameters.dti2 * DTEF[i] + 1.))
        VH[i] = A[i] * VHP[i]
        VHP[i] = (pom_bfm_parameters.dti2 * (- (SPROD[i] + E3 * BPROD[i]) * vertical_grid.length_scale[i] * E1)
                  + C[i] * VHP[i - 1] - kinetic_energy_times_length.backward[i]) * VHP[i]

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   SWEEP UPWARD
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    for i in range(2, pom_bfm_parameters.vertical_layers+1):
        k = pom_bfm_parameters.vertical_layers - i
        kinetic_energy_times_length.forward[k] = VH[k] * kinetic_energy_times_length.forward[k + 1] + VHP[k]

    for i in range(1, pom_bfm_parameters.vertical_layers - 1):
        if kinetic_energy.forward[i] > SMALL or kinetic_energy_times_length.forward[i] > SMALL:
            continue
        else:
            kinetic_energy.forward[i] = SMALL
            kinetic_energy_times_length.forward[i] = SMALL

    kinetic_energy.forward[:-1] = np.abs(kinetic_energy.forward[:-1])
    kinetic_energy_times_length.forward[:-1] = np.abs(kinetic_energy_times_length.forward[:-1])
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   THE FOLLOWING SECTION SOLVES FOR KM AND KH
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    COEF1 = A2 * (1. - 6. * A1 / B1)
    COEF2 = 3. * A2 * B2 + 18. * A1 * A2
    COEF3 = A1 * (1. - 3. * C1 - 6. * A1 / B1)
    COEF4 = 18. * A1 * A1 + 9. * A1 * A2
    COEF5 = 9. * A1 * A2

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   NOTE THAT SM AND SH LIMIT TO INFINITY WHEN GH APPROACHES 0.0288
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    vertical_grid.length_scale[0] = 0.
    vertical_grid.length_scale[pom_bfm_parameters.vertical_layers - 1] = 0.
    GH[0] = 0.
    GH[pom_bfm_parameters.vertical_layers - 1] = 0.

    vertical_grid.length_scale[1:-1] = kinetic_energy_times_length.forward[1:-1] / kinetic_energy.forward[1:-1]
    GH[1:-1] = np.power(vertical_grid.length_scale[1:-1],2) / kinetic_energy.forward[1:-1] * BOYGR[1:-1]
    for i in range(0, pom_bfm_parameters.vertical_layers):
        GH[i] = min(GH[i], .028)
    SH = COEF1 / (1. - COEF2 * GH)
    SM = COEF3 + SH * COEF4 * GH
    SM = SM / (1. - COEF5 * GH)

    KN = vertical_grid.length_scale * np.sqrt(np.abs(kinetic_energy.current))
    diffusion.kinetic_energy = 0.5 * (KN * 0.41 * SM + diffusion.kinetic_energy)
    diffusion.momentum = 0.5 * (KN * SM + diffusion.momentum)
    diffusion.tracers = 0.5 * (KN * SH + diffusion.tracers)

    return kinetic_energy, kinetic_energy_times_length, diffusion, vertical_grid


def meridional_velocity_profile(pom_bfm_parameters, vertical_grid, wind_stress, bottom_stress, diffusion, meridional_velocity):
    """ 
    Description: Calculates meridional (V) velocity profile
                 Solves for the equation:    dti2 * (KM * V')' - V = -VB
    
    :return: data array for meridional velocity profile
    """
    A = np.zeros(pom_bfm_parameters.vertical_layers)
    C = np.zeros(pom_bfm_parameters.vertical_layers)
    VH = np.zeros(pom_bfm_parameters.vertical_layers)
    VHP = np.zeros(pom_bfm_parameters.vertical_layers)
  
    A[:-2] = -pom_bfm_parameters.dti2 * (diffusion.momentum[1:-1] + pom_bfm_parameters.umol) / \
                (vertical_grid.vertical_spacing[:-2] * vertical_grid.vertical_spacing_staggered[:-2] * pom_bfm_parameters.h * pom_bfm_parameters.h)
    C[1:-1] = -pom_bfm_parameters.dti2 * (diffusion.momentum[1:-1] + pom_bfm_parameters.umol) / \
            (vertical_grid.vertical_spacing[1:-1] * vertical_grid.vertical_spacing_staggered[:-2] * pom_bfm_parameters.h * pom_bfm_parameters.h)

    VH[0] = A[0] / (A[0] - 1.)
    VHP[0] = (-pom_bfm_parameters.dti2 * wind_stress.meridional / (-vertical_grid.vertical_spacing[0] * pom_bfm_parameters.h) - meridional_velocity.forward[0]) / (A[0] - 1.)

    # 98 CONTINUE

    for i in range(1, pom_bfm_parameters.vertical_layers - 2):
        VHP[i] = 1. / (A[i] + C[i] * (1. - VH[i - 1]) - 1.)
        VH[i] = A[i] * VHP[i]
        VHP[i] = (C[i] * VHP[i - 1] - meridional_velocity.forward[i]) * VHP[i]

    CBC = 0.0
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   TO RESTORE BOTTOM B.L. DELETE NEXT LINE
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    meridional_velocity.forward[pom_bfm_parameters.vertical_layers - 2] = (C[pom_bfm_parameters.vertical_layers - 1] * VHP[pom_bfm_parameters.vertical_layers - 3] - meridional_velocity.forward[pom_bfm_parameters.vertical_layers - 2]) / (
            CBC * pom_bfm_parameters.dti2 / (-vertical_grid.vertical_spacing[pom_bfm_parameters.vertical_layers - 2] * pom_bfm_parameters.h) - 1. - (VH[pom_bfm_parameters.vertical_layers - 3] - 1.) * C[pom_bfm_parameters.vertical_layers - 2])

    for i in range(1, pom_bfm_parameters.vertical_layers - 1):
        k = pom_bfm_parameters.vertical_layers - 1 - i
        meridional_velocity.forward[k - 1] = VH[k - 1] * meridional_velocity.forward[k] + VHP[k - 1]

    bottom_stress.meridional = -CBC * meridional_velocity.forward[pom_bfm_parameters.vertical_layers - 2]  # 92

    return meridional_velocity, bottom_stress


def mixed_layer_depth(temperature,vertical_coordinates_staggered,vertical_layers):
    mixed_layer_depth = np.zeros(vertical_layers)
    small = 1.e-06

    for i in range(0,vertical_layers-1):

        if temperature[0] > temperature[i]+0.2:
            break

        mixed_layer_depth[i] = vertical_coordinates_staggered[i] - (temperature[i] + 0.2 - temperature[0]) * \
                            (vertical_coordinates_staggered[i] - vertical_coordinates_staggered[i+1]) / \
                               (temperature[i] - temperature[i+1] + small)
        
    return mixed_layer_depth


def temperature_and_salinity_profiles(pom_bfm_parameters, vertical_grid, diffusion, property, shortwave_radiation, case):
    """ 
    Description: Solves for the conservative (temperature & salinity) and non-conservative (BFM state variables) scalars
                 Handles the surface and bottom boundary conditions
    NOTE: Conservative scalars are only calculated when the system is run in prognostic mode
    
    :return: data arrays for the 'property' (temperature, salinity, or BFM state variable)
    """
    # FLAG FOR BOUNDARY CONDITION DEFINITION
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   NBC=1: SURF. B.C. IS WFSURF+SWRAD. NO RADIATIVE PENETRATION.
    #   NBC=2; SURF. B.C. IS WFSURF. SWRAD PENETRATES WATER COLUMN.
    #   NBC=3; SURF. B.C. IS TSURF. NO SWRAD RADIATIVE PENETRATION.
    #   NBC=4; SURF. B.C. IS TSURF. SWRAD PENETRATES WATER COLUMN.
    #
    #   NOTE THAT WTSURF (=WFSURF) AND SWRAD ARE NEGATIVE VALUES WHEN FLUX IS "IN" THE WATER COLUMN
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # FLAG FOR JERLOV WATER TYPE CHOICE
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   JERLOV WATER TYPE CHOICE IS RELEVANT ONLY WHEN NBC = 2 OR NBC = 4.
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    if case == 'Temperature':
        nbc  = pom_bfm_parameters.nbct
        umol = pom_bfm_parameters.umolt
    elif case == 'Salinity':
        nbc  = pom_bfm_parameters.nbcs
        umol = pom_bfm_parameters.umols
    elif case == 'BFM':
        nbc  = pom_bfm_parameters.nbcbfm
        umol = pom_bfm_parameters.umolbfm

    A = np.zeros(pom_bfm_parameters.vertical_layers)
    C = np.zeros(pom_bfm_parameters.vertical_layers)
    VH = np.zeros(pom_bfm_parameters.vertical_layers)
    VHP = np.zeros(pom_bfm_parameters.vertical_layers)

    # SW PROFILE
    vertical_radiation_profile = np.zeros(pom_bfm_parameters.vertical_layers)

    # IRRADIANCE PARAMETERS AFTER PAULSON & SIMPSON JPO 1977, 952-956
    RP = [0.58, 0.62, 0.67, 0.77, 0.78]
    AD1 = [0.35, 0.60, 1.00, 1.50, 1.40]
    AD2 = [23.00, 20.00, 17.00, 14.00, 7.90]

    # JERLOV WATER TYPES
    # NTP         = 1           2            3           4          5
    # JERLOV TYPE = I           IA           IB          II         III

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   START COMPUTATION OF VERTICAL PROFILE
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # for i in range(1, pom_bfm_parameters.vertical_layers - 1):
    #     A[i - 1] = -pom_bfm_parameters.dti2 * (diffusion.tracers[i] + umol) / (vertical_grid.vertical_spacing[i - 1] * vertical_grid.vertical_spacing_staggered[i - 1] * pom_bfm_parameters.h * pom_bfm_parameters.h)
    #     C[i] = -pom_bfm_parameters.dti2 * (diffusion.tracers[i] + umol) / (vertical_grid.vertical_spacing[i] * vertical_grid.vertical_spacing_staggered[i - 1] * pom_bfm_parameters.h * pom_bfm_parameters.h)

    A[:-2] = -pom_bfm_parameters.dti2 * (diffusion.tracers[1:-1] + umol) / (vertical_grid.vertical_spacing[:-2] * vertical_grid.vertical_spacing_staggered[:-2] * pom_bfm_parameters.h * pom_bfm_parameters.h)
    C[1:-1] = -pom_bfm_parameters.dti2 * (diffusion.tracers[1:-1] + umol) / (vertical_grid.vertical_spacing[1:-1] * vertical_grid.vertical_spacing_staggered[:-2] * pom_bfm_parameters.h * pom_bfm_parameters.h)

    vertical_radiation_profile[:] = 0.

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   SURFACE BOUNDARY CONDITION
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # *** PENETRATIVE RADIATION CALCULATION. AT THE BOTTOM ANY UNATTENUATED IS DEPOSITED IN THE BOTTOM LAYER.

    if nbc == 1:
        VH[0] = A[0] / (A[0] - 1.)
        VHP[0] = -pom_bfm_parameters.dti2 * (property.surface_flux + shortwave_radiation) / (-vertical_grid.vertical_spacing[0] * pom_bfm_parameters.h) - property.forward[0]
        VHP[0] = VHP[0] / (A[0] - 1.)

    elif nbc == 2:
        vertical_radiation_profile[:] = shortwave_radiation * (RP[pom_bfm_parameters.ntp] * np.exp(vertical_grid.vertical_coordinates[:] * pom_bfm_parameters.h / AD1[pom_bfm_parameters.ntp]) + (1. - RP[pom_bfm_parameters.ntp] * np.exp(vertical_grid.vertical_coordinates[:] * pom_bfm_parameters.h / AD2[pom_bfm_parameters.ntp])))  # ***
        vertical_radiation_profile[pom_bfm_parameters.vertical_layers - 1] = 0.

        VH[0] = A[0] / (A[0] - 1.)
        VHP[0] = pom_bfm_parameters.dti2 * (property.surface_flux + vertical_radiation_profile[0] - vertical_radiation_profile[1]) / (vertical_grid.vertical_spacing[0] * pom_bfm_parameters.h) - property.forward[0]
        VHP[0] = VHP[0] / (A[0] - 1.)

    elif nbc == 3:
        VH[0] = 0.
        VHP[0] = property.surface_value

    elif nbc == 4:
        vertical_radiation_profile[:] = shortwave_radiation * (RP[pom_bfm_parameters.ntp] * np.exp(vertical_grid.vertical_coordinates[:] * pom_bfm_parameters.h / AD1[pom_bfm_parameters.ntp]) + (1. - RP[pom_bfm_parameters.ntp] * np.exp(vertical_grid.vertical_coordinates[:] * pom_bfm_parameters.h / AD2[pom_bfm_parameters.ntp])))  # ***
        vertical_radiation_profile[pom_bfm_parameters.vertical_layers - 1] = 0.

        VH[0] = 0.
        VHP[0] = property.surface_value

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   THE FOLLOWING SECTION SOLVES THE EQUATION
    #   DT2*(KH*FF')' -FF = -FB
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    for i in range(1, pom_bfm_parameters.vertical_layers - 2):
        VHP[i] = 1 / (A[i] + C[i] * (1 - VH[i - 1]) - 1)
        VH[i] = A[i] * VHP[i]
        VHP[i] = (C[i] * VHP[i - 1] - property.forward[i] + pom_bfm_parameters.dti2 * (vertical_radiation_profile[i] - vertical_radiation_profile[i + 1]) / (pom_bfm_parameters.h * vertical_grid.vertical_spacing[i])) * VHP[i]

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   APPLY A NON ADIABATIC BOTTOM BOUNDARY CONDITION
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    property.forward[pom_bfm_parameters.vertical_layers - 2] = (C[pom_bfm_parameters.vertical_layers - 2] * VHP[pom_bfm_parameters.vertical_layers - 3] - property.forward[pom_bfm_parameters.vertical_layers - 2] \
                                                                 + (property.bottom_flux * pom_bfm_parameters.dti2 / (vertical_grid.vertical_spacing[pom_bfm_parameters.vertical_layers - 2] * pom_bfm_parameters.h))) \
                                                                / (C[pom_bfm_parameters.vertical_layers - 2] * (1 - VH[pom_bfm_parameters.vertical_layers - 3]) - 1)
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   APPLY A NON ADIABATIC BOTTOM BOUNDARY CONDITION
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    for i in range(3, pom_bfm_parameters.vertical_layers+1):
        k = (pom_bfm_parameters.vertical_layers) - i
        property.forward[k] = VH[k] * property.forward[k+1] + VHP[k]


    return property


def zonal_velocity_profile(pom_bfm_parameters, vertical_grid, wind_stress, bottom_stress, diffusion, zonal_velocity):
    """ 
    Description: Calculates zonal (U) velocity profile
                 Solves for the equation:    dti2 * (KM * U')' - U = -UB
    
    :return: data array for zonal velocity profile
    """
    A = np.zeros(pom_bfm_parameters.vertical_layers)
    C = np.zeros(pom_bfm_parameters.vertical_layers)
    VH = np.zeros(pom_bfm_parameters.vertical_layers)
    VHP = np.zeros(pom_bfm_parameters.vertical_layers)

    A[:-2] = -pom_bfm_parameters.dti2 * (diffusion.momentum[1:-1] + pom_bfm_parameters.umol) / \
                (vertical_grid.vertical_spacing[:-2] * vertical_grid.vertical_spacing_staggered[:-2] * pom_bfm_parameters.h * pom_bfm_parameters.h)
    C[1:-1] = -pom_bfm_parameters.dti2 * (diffusion.momentum[1:-1] + pom_bfm_parameters.umol) / \
               (vertical_grid.vertical_spacing[1:-1] * vertical_grid.vertical_spacing_staggered[:-2] * pom_bfm_parameters.h * pom_bfm_parameters.h)
        

    VH[0] = A[0] / (A[0] - 1.)
    VHP[0] = (-pom_bfm_parameters.dti2 * wind_stress.zonal / (-vertical_grid.vertical_spacing[0] * pom_bfm_parameters.h) - zonal_velocity.forward[0]) / (A[0] - 1.)

    for i in range(1, pom_bfm_parameters.vertical_layers - 2):
        VHP[i] = 1. / (A[i] + C[i] * (1. - VH[i - 1]) - 1.)
        VH[i] = A[i] * VHP[i]
        VHP[i] = (C[i] * VHP[i - 1] - zonal_velocity.forward[i]) * VHP[i]

    VH[0] = A[0] / (A[0] - 1.)
    VHP[0] = (-pom_bfm_parameters.dti2 * wind_stress.zonal / (-vertical_grid.vertical_spacing[0] * pom_bfm_parameters.h) - zonal_velocity.forward[0]) / (A[0] - 1.)

    CBC = 0.0
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    zonal_velocity.forward[pom_bfm_parameters.vertical_layers - 2] = (C[pom_bfm_parameters.vertical_layers - 2] * VHP[pom_bfm_parameters.vertical_layers - 3] - zonal_velocity.forward[pom_bfm_parameters.vertical_layers - 2]) / (
            CBC * pom_bfm_parameters.dti2 / (-vertical_grid.vertical_spacing[pom_bfm_parameters.vertical_layers - 2] * pom_bfm_parameters.h) - 1. - (VH[pom_bfm_parameters.vertical_layers - 3] - 1.) * C[pom_bfm_parameters.vertical_layers - 2])
    for i in range(1, pom_bfm_parameters.vertical_layers - 1):
        k = pom_bfm_parameters.vertical_layers - 1 - i
        zonal_velocity.forward[k - 1] = VH[k - 1] * zonal_velocity.forward[k] + VHP[k - 1]
    bottom_stress.zonal = -CBC * zonal_velocity.forward[pom_bfm_parameters.vertical_layers - 2]  # 92

    return zonal_velocity, bottom_stress

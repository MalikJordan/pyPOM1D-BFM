from math import ceil
import numpy as np
import os
current_path = os.getcwd()

class PomBfm:
    def __init__(self):
        self.pom_only = False               # True if just pom, False if pom_bfm

        # Simulation
        self.dti = 360.0                    # timestep [s]
        self.dti2 = 2 * self.dti            # twice the timestep [s]
        self.days = 730                     # length of simulation [days]
        # self.days = 10                     # length of simulation [days]
        self.months = ceil(self.days/30)    # total months in simulation (for data averaging)
        self.savef = 1                      # output averaging and saving frequency [days]
        self.sec_per_day = 86400
        self.dayi = 1./self.sec_per_day
        self.iterations_needed = int(self.days * self.sec_per_day / self.dti)
        
        # Water Column 
        self.h = 150.0                      # depth [m]
        self.upperh = 5.0                   # depth where lateral advection starts [m]
        self.vertical_layers = 151          # number of layers in water column
        self.num_boxes = self.vertical_layers - 1
        self.kl1 = 2                        # surface logarithmic layers distribution
        self.kl2 = 150                      # bottom logarithmic layers distribution
        
        # General
        self.water_specific_heat_times_density = 4.187e06
        self.earth_angular_velocity = 7.29e-05
        self.alat = 45.0                    # latitude [degrees]
        self.idiagn = 1                     # switch between prognostic (idiagn = 0) and diagnostic (idiagn = 1) mode
        self.smoth = 0.1                    # parameter for hasselin filter
        self.ihotst = 0                     # switch for cold start (ihotst = 0) and hot start, ie reading restart (ihotst = 1)
        self.coriolis = 2. * self.earth_angular_velocity * np.sin(self.alat * 2. * np.pi / 360.)
        
        # Relaxation Velocities [m/d]
        self.nrt_o2o = 0.06                 # oxygen
        self.nrt_n1p = 0.06                 # phosphate
        self.nrt_n3n = 0.06                 # nitrate
        self.nrt_n4n = 0.05                 # ammonium

        # Flags
        self.nbct = 2                       # temperature boundary conditions
        self.nbcs = 1                       # salinity boundary conditions
        self.nbcbfm = 1                     # bfm boundary conditions
        self.ntp = 2                        # jerlov water type ( 1 = I, 2 = IA, 3 = IB, 4 = II, 5 = III)

        # Background Diffusions [m2/s]
        self.umol = 1.e-06                  # general
        self.umolt = 1.e-07                 # temperature
        self.umols = 1.3e-07                # salalinity
        self.umolbfm = 1.e-04               # bfm

        # Relaxation Times [s]
        self.trt = 0                        # lateral temperature advection
        self.srt = 1                        # lateral salinity advection
        self.ssrt = 5.68                    # surface salinity flux


class PomInputFiles:
    def __init__(self):
        self.wind_stress = current_path + '/inputs/POM_BFM17/monthly_surf_wind_stress_bermuda_killworth2.da'
        self.surface_salinity = current_path + '/inputs/POM_BFM17/monthly_surf_salt_bermuda_150m_killworth2.da'
        self.shortwave_solar_radiation = current_path + '/inputs/POM_BFM17/monthly_surf_qs_bermuda_killworth2.da'
        self.inorganic_suspended_matter = current_path + '/inputs/POM_BFM17/monthly_clima_ISM_150m_bermuda_killworth.da'
        self.salinity_vertical_profile = current_path + '/inputs/POM_BFM17/monthly_clima_salt_150m_bermuda_killworth2.da'
        self.temperature_vertical_profile = current_path + '/inputs/POM_BFM17/monthly_clima_temp_150m_bermuda_killworth2.da'
        self.general_circulation_w_velocity = current_path + '/inputs/POM_BFM17/monthly_clima_w_150m_bermuda_ekman.da'
        self.intermediate_eddy_w_velocity_1 = current_path + '/inputs/POM_BFM17/bimonthly_random_eddy_w_150m_bermuda_norm1.da'
        self.intermediate_eddy_w_velocity_2 = current_path + '/inputs/POM_BFM17/bimonthly_random_eddy_w_150m_bermuda_norm2.da'
        self.salinity_initial_conditions = current_path + '/inputs/POM_BFM17/init_prof_S_150m_bermuda_killworth2.da'
        self.temperature_initial_conditions = current_path + '/inputs/POM_BFM17/init_prof_T_150m_bermuda_killworth2.da'
        self.heat_flux_loss = current_path + '/inputs/POM_BFM17/monthly_surf_rad_bermuda_killworth2.da'
        self.surface_nutrients = current_path + '/inputs/POM_BFM17/NutrientsARPAOGS.da'
        self.bottom_nutrients = current_path + '/inputs/POM_BFM17/monthly_bott_nut_bermuda_150m_killworth.da'
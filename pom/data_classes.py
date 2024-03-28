import numpy as np

class Diffusion:
    def __init__(self,vertical_layers):
        self.tracers = np.zeros(vertical_layers)
        self.momentum = np.zeros(vertical_layers)
        self.kinetic_energy = np.zeros(vertical_layers)

class ForcingManagerCounters:
    def __init__(self):
        self.day_counter = 0
        self.day_interpolator = 0
        self.ratio_day = 0
        self.month_counter = 0
        self.month_interpolator = 0
        self.ratio_month = 0
        self.timesteps_per_day = 0
        self.timesteps_per_month = 0

class KineticEnergyAndVelocity:
    def __init__(self,initial_value,vertical_layers):
        self.current = initial_value * np.ones(vertical_layers)
        self.forward = initial_value * np.ones(vertical_layers)
        self.backward = initial_value * np.ones(vertical_layers)

class MonthlyForcing:
    def __init__(self,vertical_layers):
        self.sclim = np.zeros(vertical_layers)
        self.tclim = np.zeros(vertical_layers)
        self.wclim = np.zeros(vertical_layers)
        self.weddy1 = np.zeros(vertical_layers)
        self.weddy2 = np.zeros(vertical_layers)
        self.ism = np.zeros(vertical_layers-1)
        self.wsu = 0
        self.wsv = 0
        self.swrad = 0
        self.wtsurf = 0
        self.qcorr = 0
        self.NO3_s = 0
        self.NH4_s = 0
        self.PO4_s = 0
        self.SIO4_s = 0
        self.O2_b = 0
        self.NO3_b = 0
        self.PO4_b = 0
        self.PON_b = 0

class Nutrients:
    def __init__(self):
        self.NO3surf = 0
        self.NH4surf = 0
        self.PO4surf = 0
        self.SIO4surf = 0
        self.O2bott = 0
        self.NO3bott = 0
        self.PO4bott = 0
        self.PONbott_grad = 0

class Stresses:
    def __init__(self):
        self.zonal = 0
        self.meridional = 0

class TemperatureAndSalinity:
    def __init__(self,vertical_layers):
        self.current = np.zeros(vertical_layers)
        self.forward = np.zeros(vertical_layers)
        self.backward = np.zeros(vertical_layers)
        self.interpolated = np.zeros(vertical_layers)
        self.surface_value = 0
        self.surface_flux = 0
        self.bottom_flux = 0
        self.lateral_advection = np.zeros(vertical_layers)

class VerticalGrid:
    def __init__(self,vertical_layers):
        self.length_scale = np.ones(vertical_layers)
        self.vertical_coordinates = np.zeros(vertical_layers)
        self.vertical_coordinates_staggered = np.zeros(vertical_layers)
        self.vertical_spacing = np.zeros(vertical_layers)
        self.vertical_spacing_staggered = np.zeros(vertical_layers)
        self.vertical_spacing_reciprocal = np.zeros(vertical_layers)

    def coordinate_system(self,surface_layers_with_log_distribution,bottom_layers_with_log_distribution,vertical_layers):
        surface_logspace_layers = surface_layers_with_log_distribution - 2.
        bottom_logspace_layers = vertical_layers - bottom_layers_with_log_distribution - 1.

        BB = (bottom_layers_with_log_distribution - surface_layers_with_log_distribution) + 4.
        CC = surface_layers_with_log_distribution - 2.
        initial_spacing = 2. / BB / np.exp(.693147 * (surface_layers_with_log_distribution - 2))

        self.vertical_coordinates_staggered[0] = -0.5 * initial_spacing

        for i in range(1,int(surface_layers_with_log_distribution)-1):
            self.vertical_coordinates[i-1] = -initial_spacing * 2**(i-2)
            self.vertical_coordinates_staggered[i-1] = -initial_spacing * 2**(i-1.5)

        for i in range(int(surface_layers_with_log_distribution)-1,vertical_layers+1):
            self.vertical_coordinates[i-1]  = -(i - surface_logspace_layers) / (bottom_layers_with_log_distribution
                                                                       - surface_layers_with_log_distribution + 4.)
            self.vertical_coordinates_staggered[i-1] = -(i - surface_logspace_layers + 0.5) / (bottom_layers_with_log_distribution
                                                                                        - surface_layers_with_log_distribution + 4.)
            
        self.vertical_spacing[:-1] = self.vertical_coordinates[:-1] - self.vertical_coordinates[1:]
        self.vertical_spacing_staggered[:-1] = self.vertical_coordinates_staggered[:-1] - self.vertical_coordinates_staggered[1:]

        self.vertical_spacing[-1] = 1.e-06  # small value to avoid division by zero for vertical_spacing_reciprocal
        self.vertical_spacing_reciprocal = 1. / self.vertical_spacing   # take reciprocal
        self.vertical_spacing[-1] = 0.      # correct value

        self.length_scale[0] = 0.
        self.length_scale[-1] = 0.
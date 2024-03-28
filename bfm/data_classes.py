import numpy as np

class BfmModel:
    """
    Indicates model used in coupled POM-BFM
    """
    def __init__(self):
        self.bfm50 = True
        self.bfm36 = False
        self.bfm35 = False
        self.bfm34 = False
        self.bfm23 = False
        self.bfm1  = False


class BfmStateVariables:
    def __init__(self,vertical_layers):
        self.current = np.zeros(vertical_layers)
        self.forward = np.zeros(vertical_layers)
        self.backward = np.zeros(vertical_layers)
        self.surface_value = 0.
        self.surface_flux = 0.
        self.bottom_flux = 0.


class BfmPhysicalVariables:
    def __init__(self, num_boxes,ph_initial):
        self.temperature = np.zeros(num_boxes)
        self.salinity = np.zeros(num_boxes)
        self.density = np.zeros(num_boxes)
        self.suspended_matter = np.zeros(num_boxes)
        self.depth = np.zeros(num_boxes)
        self.irradiance = np.zeros(num_boxes)
        self.vertical_extinction = np.zeros(num_boxes)
        self.wind = 0.
        self.wgen = np.zeros(num_boxes)
        self.weddy = np.zeros(num_boxes)
        self.detritus_sedimentation = np.zeros(num_boxes)
        self.phyto_sedimentation = np.zeros((num_boxes,4))
        self.pH = ph_initial


class NutrientData:
    def __init__(self):
        self.NO3surf = 0.
        self.NH4surf = 0.
        self.PO4surf = 0.
        self.SIO4surf = 0.
        self.O2bott = 0.
        self.NO3bott = 0.
        self.PO4bott = 0.
        self.PONbott_grad = 0.


class OutputAverages:
    """
    Definition: Initializes matrix for daily and monthly averages of fields of interests
    
    Fields of interst: Chlorophyll-a, Oxygen, Nitrate, Phosphate, Particulate Organic Nitrogen, Net Primary Production, Dissolved Inorganic Carbon
    """
    def __init__(self,days,months,num_boxes):
        self.count = 0.
        self.day = 0
        self.month = 0
        self.daily_ave = np.zeros((7,num_boxes,days))
        self.monthly_ave = np.zeros((7,num_boxes,months))
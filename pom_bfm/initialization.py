import numpy as np
from bfm.parameters import Bfm, BfmInputFiles
from pom.parameters import PomBfm
from os import path

bfm_parameters = Bfm()
input_files = BfmInputFiles()
pom_bfm_parameters = PomBfm()

def initialize_bfm_in_pom(vertical_grid):
    """
    Description: Handles the BFM setup and state variable initialization for the coupled 1D system
    """
    # Set the thickness of the vertical_layers-1 layers
    depth = np.zeros(pom_bfm_parameters.vertical_layers)
    depth[:] = vertical_grid.vertical_spacing[:] * pom_bfm_parameters.h

    # Allocate and initialize additional integration arrays
    d3state = np.zeros((pom_bfm_parameters.num_boxes,bfm_parameters.num_d3_box_states))
    d3stateb = np.zeros((pom_bfm_parameters.num_boxes,bfm_parameters.num_d3_box_states))

    # Define initial conditions
    bfm_variables1 = set_initial_conditions_bfm50()
    bfm_variables2 = set_initial_conditions_bfm50()

    # Initialize prior time step for leapfrog
    d3state = bfm_variables1[:pom_bfm_parameters.num_boxes,:]
    d3stateb = bfm_variables2[:pom_bfm_parameters.num_boxes,:]

    return d3state, d3stateb


def read_bfm_input():
    """
    Description: Handles the reading of initial data files for BFM variables
    """
    # Phytoplankton carbon initial condition
    if path.exists(input_files.phytoc):
        p2c = np.fromfile(input_files.phytoc)

    # Zooplankton carbon initial condition
    if path.exists(input_files.zooc):
        z5c = np.fromfile(input_files.zooc)

    # Particulate organic carbon initial condition
    if path.exists(input_files.poc):
        r6c = np.fromfile(input_files.poc)

    # Dissolved organic carbon initial condition
    if path.exists(input_files.doc):
        r1c = np.fromfile(input_files.doc)

    # Phosphate initial condition
    if path.exists(input_files.phos):
        n1p = np.fromfile(input_files.phos)

    # Nitrate initial condition
    if path.exists(input_files.nit):
        n3n = np.fromfile(input_files.nit)

    # Ammonium initial condition
    if path.exists(input_files.am):
        n4n = np.fromfile(input_files.am)

    # Oxygen initial condition
    if path.exists(input_files.oxy):
        o2o = np.fromfile(input_files.oxy)

    return p2c, z5c, r6c, r1c, n1p, n3n, n4n, o2o


def set_initial_conditions_bfm50():
    """
    Description: Initializes BFM state variable matrix using initial variables defined by namelist data and funciton 'read_bfm_input()'
    """
    # Local Variables
    p_nRc = 0.0126
    p_pRc = 0.7862e-3
    p_sRc = 0.0118
    p_iRc = 1./25.

    # Definition of general pelagic state variables: Pelagic Fases
    p2c, z5c, r6c, r1c, n1p, n3n, n4n, o2o = read_bfm_input()
    n4n[:] = 0. # From get_IC.f90 --> N4n(k) = 0.0

    o3c = bfm_parameters.o3c0 * np.ones(pom_bfm_parameters.vertical_layers)
    o3h = bfm_parameters.o3h0 * np.ones(pom_bfm_parameters.vertical_layers)

    # Pelagic nutrients (mMol / m3)
    n5s = bfm_parameters.n5s0 * np.ones(pom_bfm_parameters.vertical_layers)
    o4n = bfm_parameters.o4n0 * np.ones(pom_bfm_parameters.vertical_layers)
    n6r = bfm_parameters.n6r0 * np.ones(pom_bfm_parameters.vertical_layers)

    # Pelagic detritus (respectively mg C/m3 mMol N/m3 mMol P/m3)
    r6n = r6c * p_nRc
    r6p = r6c * p_pRc
    r6s = r6c * p_sRc
    r2c = bfm_parameters.r2c0 * np.ones(pom_bfm_parameters.vertical_layers)
    r3c = bfm_parameters.r3c0 * np.ones(pom_bfm_parameters.vertical_layers)

    # Dissolved organic matter
    r1n = r1c * p_nRc * 0.5
    r1p = r1c * p_pRc * 0.5

    # State variables for phytoplankton model
    # Pelagic diatoms (respectively mg C/m3 mMol N/m3 mMol P/m3)
    p1c = 0.05 * p2c
    p1n = p1c * p_nRc
    p1p = p1c * p_pRc
    p1s = p1c * p_sRc
    p1l = p1c * p_iRc

    # Picophytoplankton (respectively mg C/m3 mMol N/m3 mMol P/m3)
    p3c = 0.1 * p2c
    p3n = p3c * p_nRc
    p3p = p3c * p_pRc
    p3l = p3c * p_iRc

    # Large phytoplankton (respectively mg C/m3 mMol N/m3 mMol P/m3)
    p4c = 0.05 * p2c
    p4n = p4c * p_nRc
    p4p = p4c * p_pRc
    p4l = p4c * p_iRc
    
    # Pelagic flagellates (respectively mg C/m3 mMol N/m3 mMol P/m3)
    p2c = 0.8 * p2c
    p2n = p2c * p_nRc
    p2p = p2c * p_pRc
    p2l = p2c * p_iRc
    
    # State variables for mesozooplankton model
    # Carnivorous mesozooplankton ( mg C/m3 )
    z3c = bfm_parameters.z3c0 * np.ones(pom_bfm_parameters.vertical_layers)
    z3n = z3c * p_nRc
    z3p = z3c * p_pRc

    # Omnivorous mesozooplankton ( mg C/m3 )
    z4c = bfm_parameters.z4c0 * np.ones(pom_bfm_parameters.vertical_layers)
    z4n = z4c * p_nRc
    z4p = z4c * p_pRc

    # Heterotrophic flagellates (respectively mg C/m3 mMol N/m3 mMol P/m3)
    z6c = 0.1 * z5c
    z6n = z6c * p_nRc
    z6p = z6c * p_pRc
    
    # State variables for microzooplankton model
    # Pelagic microzooplankton  (respectively mg C/m3 mMol N/m3 mMol P/m3)
    z5c = 0.9 * z5c
    z5n = z5c * p_nRc
    z5p = z5c * p_pRc
    
    # State variables for pelagic bacteria model B1
    # Pelagic bacteria (respectively mg C/m3 mMol N/m3 mMol P/m3)
    b1c = bfm_parameters.b1c0 * np.ones(pom_bfm_parameters.vertical_layers)
    b1n = b1c * p_nRc
    b1p = b1c * p_pRc
    
    # Fill matrix with bfm variable data
    bfm50_variables = np.zeros((pom_bfm_parameters.vertical_layers,bfm_parameters.num_d3_box_states))

    bfm50_variables[:,0] = o2o[:]
    bfm50_variables[:,1] = n1p[:]
    bfm50_variables[:,2] = n3n[:]
    bfm50_variables[:,3] = n4n[:]
    bfm50_variables[:,4] = o4n[:]
    bfm50_variables[:,5] = n5s[:]
    bfm50_variables[:,6] = n6r[:]

    bfm50_variables[:,7] = b1c[:]
    bfm50_variables[:,8] = b1n[:]
    bfm50_variables[:,9] = b1p[:]
 
    bfm50_variables[:,10] = p1c[:]
    bfm50_variables[:,11] = p1n[:]
    bfm50_variables[:,12] = p1p[:]
    bfm50_variables[:,13] = p1l[:]
    bfm50_variables[:,14] = p1s[:]
 
    bfm50_variables[:,15] = p2c[:]
    bfm50_variables[:,16] = p2n[:]
    bfm50_variables[:,17] = p2p[:]
    bfm50_variables[:,18] = p2l[:]

    bfm50_variables[:,19] = p3c[:]
    bfm50_variables[:,20] = p3n[:]
    bfm50_variables[:,21] = p3p[:]
    bfm50_variables[:,22] = p3l[:]

    bfm50_variables[:,23] = p4c[:]
    bfm50_variables[:,24] = p4n[:]
    bfm50_variables[:,25] = p4p[:]
    bfm50_variables[:,26] = p4l[:]

    bfm50_variables[:,27] = z3c[:]
    bfm50_variables[:,28] = z3n[:]
    bfm50_variables[:,29] = z3p[:]

    bfm50_variables[:,30] = z4c[:]
    bfm50_variables[:,31] = z4n[:]
    bfm50_variables[:,32] = z4p[:]
 
    bfm50_variables[:,33] = z5c[:]
    bfm50_variables[:,34] = z5n[:]
    bfm50_variables[:,35] = z5p[:]
 
    bfm50_variables[:,36] = z6c[:]
    bfm50_variables[:,37] = z6n[:]
    bfm50_variables[:,38] = z6p[:]

    bfm50_variables[:,39] = r1c[:]
    bfm50_variables[:,40] = r1n[:]
    bfm50_variables[:,41] = r1p[:]
 
    bfm50_variables[:,42] = r2c[:]
    bfm50_variables[:,43] = r3c[:]

    bfm50_variables[:,44] = r6c[:]
    bfm50_variables[:,45] = r6n[:]
    bfm50_variables[:,46] = r6p[:]
    bfm50_variables[:,47] = r6s[:]

    bfm50_variables[:,48] = o3c[:]
    bfm50_variables[:,49] = o3h[:]

    return bfm50_variables

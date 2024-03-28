import numpy as np
from bfm.bfm50.Functions.other_functions import eTq_vector, get_concentration_ratio

def microzoo_eqns(d3state, microzoo_parameters, constant_parameters, environmental_parameters, zc, zn, zp, i_c, i_n, i_p, temp, o2o):
    """ Calculates the micorzooplankton (Z5 & Z6) terms needed for the zooplankton biological rate equations
        Equations come from the BFM user manual and the fortran code MicroZoo.F90
    """

    num_boxes = d3state.shape[0]

    # Concentration ratios
    zn_zc = get_concentration_ratio(zn, zc, constant_parameters.p_small)
    zp_zc = get_concentration_ratio(zp, zc, constant_parameters.p_small)

    # Temperature regulating factor (et)
    fTZ = eTq_vector(temp, environmental_parameters.basetemp, environmental_parameters.q10z)
    
    # Oxygen dependent regulation factor (eO2)
    fZO = np.zeros(num_boxes)
    for i in range(0,num_boxes):
        fZO[i] = min(1.0, (o2o[i]/(o2o[i] + microzoo_parameters.z_o2o)))
    
    #---------------------- Microzooplankton Respiration ----------------------
    # Zooplankton total repiration rate (eqn. 2.4.8, and matches fortran code)
    rrac = i_c*(1.0 - microzoo_parameters.etaZ - microzoo_parameters.betaZ)
    rrsc = microzoo_parameters.bZ*fTZ*zc
    dZcdt_rsp_o3c = rrac + rrsc
    
    #------------- Microzooplankton mortality and activity excretion ----------
    # From fortran code MesoZoo.F90 lines 327-331
    rdc = ((1.0 - fZO)*microzoo_parameters.d_ZO + microzoo_parameters.d_Z)*zc
    reac = i_c*(1.0 - microzoo_parameters.etaZ)*microzoo_parameters.betaZ
    rric = reac + rdc
    dZcdt_rel_r1c = rric*constant_parameters.epsilon_c
    dZcdt_rel_r6c = rric*(1.0 - constant_parameters.epsilon_c)    

    #------------------- Microzooplankton nutrient dynamics -------------------
    # Organic Nitrogen dynamics (from fortran code) [mmol N m^-3 s^-1]
    rrin = i_n*microzoo_parameters.betaZ + rdc*zn_zc
    dZndt_rel_r1n = rrin*constant_parameters.epsilon_n
    dZndt_rel_r6n = rrin - dZndt_rel_r1n

    # Organic Phosphorus dynamics (from fortran code) [mmol P m^-3 s^-1]
    rrip = i_p*microzoo_parameters.betaZ + rdc*zp_zc
    dZpdt_rel_r1p = rrip*constant_parameters.epsilon_p
    dZpdt_rel_r6p = rrip - dZpdt_rel_r1p

    #--------------- Microzooplankton Dissolved nutrient dynamics -------------     
    # Equations from fortran code (MicroZoo.F90 line 368-371)
    runc = np.zeros(num_boxes)
    runn = np.zeros(num_boxes)
    runp = np.zeros(num_boxes)
    dZpdt_rel_n1p = np.zeros(num_boxes)
    dZndt_rel_n4n = np.zeros(num_boxes)
    for i in range(0,num_boxes):
        runc[i] = max(0.0, i_c[i]*(1.0 - microzoo_parameters.betaZ)-rrac[i])
        runn[i] = max(0.0, i_n[i]*(1.0 - microzoo_parameters.betaZ) + rrsc[i]*zn_zc[i])
        runp[i] = max(0.0, i_p[i]*(1.0 - microzoo_parameters.betaZ) + rrsc[i]*zp_zc[i])
        dZpdt_rel_n1p[i] = max(0.0, runp[i]/(constant_parameters.p_small + runc[i]) - 0.0007862)*runc[i]   # MicroZoo.F90, rep
        dZndt_rel_n4n[i] = max(0.0, runn[i]/(constant_parameters.p_small + runc[i]) - 0.01258)*runc[i]   # MicroZoo.F90, ren

    return dZcdt_rel_r1c, dZcdt_rel_r6c, dZcdt_rsp_o3c, dZndt_rel_r1n, dZndt_rel_r6n, dZpdt_rel_r1p, dZpdt_rel_r6p, dZpdt_rel_n1p, dZndt_rel_n4n
    

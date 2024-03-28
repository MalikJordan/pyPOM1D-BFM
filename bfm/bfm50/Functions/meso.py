import numpy as np
from bfm.bfm50.Functions.other_functions import eTq_vector, get_concentration_ratio

def mesozoo_eqns(d3state, mesozoo_parameters, constant_parameters, environmental_parameters, zc, zn, zp, i_c, i_n, i_p, temp, o2o):
    """ Calculates the mesozooplankton (Z3 & Z4) terms needed for the zooplankton biological rate equations
        Equations come from the BFM user manual and the fortran code MesoZoo.F90
    """         
    
    num_boxes = d3state.shape[0]

    # Concentration ratios
    zn_zc = get_concentration_ratio(zn, zc, constant_parameters.p_small)
    zp_zc = get_concentration_ratio(zp, zc, constant_parameters.p_small)
    
    # Temperature regulating factor (from MesoZoo.F90 'et')
    fTZ = eTq_vector(temp, environmental_parameters.basetemp, environmental_parameters.q10z)
    
    # Oxygen dependent regulation factor (from MesoZoo.F90 'eo')
    fZO = np.zeros(num_boxes)
    for i in range(0,num_boxes):
        fZO[i] = (max(constant_parameters.p_small,o2o[i])**3)/(max(constant_parameters.p_small,o2o[i])**3 + mesozoo_parameters.z_o2o**3)
    
    # energy cost of ingestion
    prI = 1.0 - mesozoo_parameters.etaZ - mesozoo_parameters.betaZ
    
    # Zooplankton total repiration rate (from MesoZoo.F90 line 343 'rrc')
    dZcdt_rsp_o3c = prI*i_c + mesozoo_parameters.bZ*fTZ*zc
    
    # Specific rates of low oxygen mortality and Density dependent mortality
    # from fortran code MesoZoo.F90 lines 343-344
    rdo_c = mesozoo_parameters.d_Zdns*(1.0 - fZO)*fTZ*zc
    rd_c = mesozoo_parameters.d_Z*zc**mesozoo_parameters.gammaZ
    
    # Total egestion including pellet production (from MesoZoo.F90 line 359 - 361)
    dZcdt_rel_r6c = mesozoo_parameters.betaZ*i_c + rdo_c + rd_c
    dZndt_rel_r6n = mesozoo_parameters.betaZ*i_n + zn_zc*(rdo_c + rd_c)
    dZpdt_rel_r6p = mesozoo_parameters.betaZ*i_p + zp_zc*(rdo_c + rd_c)
    
    # Check the assimilation rate for Carbon, Nitrogen and Phosphorus
    # compute P:C and N:C ratios in the assimilation rate
    # from MesoZoo.F90 lines 371-375
    ru_c = mesozoo_parameters.etaZ*i_c
    ru_n = (mesozoo_parameters.etaZ + prI)*i_n
    ru_p = (mesozoo_parameters.etaZ + prI)*i_p
    pu_e_n = ru_n/(constant_parameters.p_small + ru_c)
    pu_e_p = ru_p/(constant_parameters.p_small + ru_c)
    
    # Eliminate the excess of the non-limiting constituent
    # Determine whether C, P or N is the limiting element and assign the value to variable limiting_nutrient
    # from MesoZoo.F90 lines 
    q_Zc = np.zeros(num_boxes)
    q_Zp = np.zeros(num_boxes)
    q_Zn = np.zeros(num_boxes)

    temp_p = pu_e_p/(zp_zc + constant_parameters.p_small)
    temp_n = pu_e_n/(zn_zc + constant_parameters.p_small)
    for i in range(0,num_boxes):
        limiting_nutrient = 'carbon'
        
        if temp_p[i]<temp_n[i] or abs(temp_p[i] - temp_n[i])<constant_parameters.p_small:
            if pu_e_p[i]<zp_zc[i]:
            # if pu_e_p[i]<mesozoo_parameters.p_qpcMEZ:
                limiting_nutrient = 'phosphorus'
        else:
            if pu_e_n[i]<zn_zc[i]:
            # if pu_e_n[i]<mesozoo_parameters.p_qncMEZ:
                limiting_nutrient = 'nitrogen'
        
        # Compute the correction terms depending on the limiting constituent
        if limiting_nutrient == 'carbon':
            q_Zc[i] = 0.0
            q_Zp[i] = max(0.0, (1.0 - mesozoo_parameters.betaZ)*i_p[i] - mesozoo_parameters.p_qpcMEZ*ru_c[i])
            q_Zn[i] = max(0.0, (1.0 - mesozoo_parameters.betaZ)*i_n[i] - mesozoo_parameters.p_qncMEZ*ru_c[i])
        elif limiting_nutrient == 'phosphorus':
            q_Zp[i] = 0.0
            q_Zc[i] = max(0.0, ru_c[i] - (1.0 - mesozoo_parameters.betaZ)*i_p[i]/mesozoo_parameters.p_qpcMEZ)
            q_Zn[i] = max(0.0, (1.0 - mesozoo_parameters.betaZ)*i_n[i] - mesozoo_parameters.p_qncMEZ*(ru_c[i] - q_Zc[i]))
        elif limiting_nutrient == 'nitrogen':
            q_Zn[i] = 0.0
            q_Zc[i] = max(0.0, ru_c[i] - (1.0 - mesozoo_parameters.betaZ)*i_n[i]/mesozoo_parameters.p_qncMEZ)
            q_Zp[i] = max(0.0, (1.0 - mesozoo_parameters.betaZ)*i_p[i] - mesozoo_parameters.p_qpcMEZ*(ru_c[i] - q_Zc[i]))

    # Nutrient remineralization basal metabolism + excess of non-limiting nutrients
    dZpdt_rel_n1p = mesozoo_parameters.bZ*fZO*fTZ*zp + q_Zp    # (from MesoZoo.F90 'rep')
    dZndt_rel_n4n = mesozoo_parameters.bZ*fZO*fTZ*zn + q_Zn    # (from MesoZoo.F90 'ren')
    
    # Fluxes to particulate organic matter 
    # Add the correction term for organic carbon release based on the limiting constituent
    x = mesozoo_parameters.betaZ*i_c + rdo_c + rd_c
    dZcdt_rel_r6c += q_Zc  # (from MesoZoo.F90 'rq6c')
    
    # mesozooplankton are assumed to have no dissolved products
    dZcdt_rel_r1c = np.zeros(num_boxes)
    dZndt_rel_r1n = np.zeros(num_boxes)
    dZpdt_rel_r1p = np.zeros(num_boxes)
    
    
    return dZcdt_rel_r1c, dZcdt_rel_r6c, dZcdt_rsp_o3c, dZndt_rel_r1n, dZndt_rel_r6n, dZpdt_rel_r1p, dZpdt_rel_r6p, dZpdt_rel_n1p, dZndt_rel_n4n

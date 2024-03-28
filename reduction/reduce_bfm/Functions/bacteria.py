import numpy as np
from bfm.bfm50.Functions.other_functions import eTq_vector, get_concentration_ratio, insw_vector

def bacteria_eqns(d3state, bacteria_parameters, constant_parameters, environmental_parameters, temper):
    """ Calculates the terms needed for the bacteria biological rate equations
        Equations come from the BFM user manual
    """
    
    num_boxes = d3state.shape[0]

    # State variables
    o2o = d3state[:,0]              # Dissolved oxygen (mg O_2 m^-3)
    n1p = d3state[:,1]              # Phosphate (mmol P m^-3)
    n4n = d3state[:,3]              # Ammonium (mmol N m^-3)
    b1c = d3state[:,7]              # Pelagic bacteria carbon (mg C m^-3)
    b1n = d3state[:,8]              # Pelagic bacteria nitrogen (mmol N m^-3)
    b1p = d3state[:,9]              # Pelagic bacteria phosphate (mmol P m^-3)
    r1c = d3state[:,39]             # Labile dissolved organic carbon (mg C m^-3)
    r1n = d3state[:,40]             # Labile dissolved organic nitrogen (mmol N m^-3)
    r1p = d3state[:,41]             # Labile dissolved organic phosphate (mmol P m^-3)
    r2c = d3state[:,42]             # Semi-labile dissolved organic carbon (mg C m^-3)
    r3c = d3state[:,43]             # Semi-refractory Dissolved Organic Carbon (mg C m^-3)
    r6c = d3state[:,44]             # Particulate organic carbon (mg C m^-3)
    r6n = d3state[:,45]             # Particulate organic nitrogen (mmol N m^-3)
    r6p = d3state[:,46]             # Particulate organic phosphate (mmol P m^-3)
    
    # concentration ratios   
    bp_bc = get_concentration_ratio(b1p, b1c, constant_parameters.p_small)
    bn_bc = get_concentration_ratio(b1n, b1c, constant_parameters.p_small)
    r1p_r1c = get_concentration_ratio(r1p, r1c, constant_parameters.p_small)
    r6p_r6c = get_concentration_ratio(r6p, r6c, constant_parameters.p_small)
    r1n_r1c = get_concentration_ratio(r1n, r1c, constant_parameters.p_small)
    r6n_r6c = get_concentration_ratio(r6n, r6c, constant_parameters.p_small)
    
    # Temperature effect on pelagic bacteria ('et')
    fTB = eTq_vector(temper, environmental_parameters.basetemp, environmental_parameters.q10b)

    # oxygen non-dimensional regulation factor[-]
    # Oxygen environment: bacteria are both aerobic and anaerobic ('eO2')
    f_B_O = np.zeros(num_boxes)
    for i in range(0,num_boxes):
        f_B_O[i] = max(constant_parameters.p_small,o2o[i])**3/(max(constant_parameters.p_small,o2o[i])**3 + bacteria_parameters.h_B_O**3)
    
    # external nutrient limitation (from PelBac.F90 'eN4n' & 'eN1p')
    f_B_n = n4n/(n4n + bacteria_parameters.h_B_n)
    f_B_p = n1p/(n1p + bacteria_parameters.h_B_p)
    
    # Bacteria mortality (lysis) process [mg C m^-3 s^-1]
    dBcdt_lys = (bacteria_parameters.d_0B*fTB + bacteria_parameters.d_B_d*b1c)*b1c  # (from PelBac.F90 'rd')
    dBcdt_lys_r1c = dBcdt_lys*constant_parameters.epsilon_c
    dBcdt_lys_r1n = dBcdt_lys*bn_bc*constant_parameters.epsilon_n
    dBcdt_lys_r1p = dBcdt_lys*bp_bc*constant_parameters.epsilon_p
    dBcdt_lys_r6c = dBcdt_lys*(1.0 - constant_parameters.epsilon_c)
    dBcdt_lys_r6n = dBcdt_lys*bn_bc*(1.0 - constant_parameters.epsilon_n)
    dBcdt_lys_r6p = dBcdt_lys*bp_bc*(1.0 - constant_parameters.epsilon_p)


    # Substrate availability
    if bacteria_parameters.bact_version==1 or bacteria_parameters.bact_version==2:
        # nutrient limitation (intracellular) (from PelBac.F90 'iN1n', 'iN1p', 'iN')
        nut_lim_n = np.zeros(num_boxes)
        nut_lim_p = np.zeros(num_boxes)
        f_B_n_P = np.zeros(num_boxes)
        for i in range(0,num_boxes):
            nut_lim_n[i] = min(1.0, max(0.0, bn_bc[i]/bacteria_parameters.n_B_opt))         # Nitrogen
            nut_lim_p[i] = min(1.0, max(0.0, bp_bc[i]/bacteria_parameters.p_B_opt))         # Phosphorus
            f_B_n_P[i] = min(nut_lim_n[i], nut_lim_p[i])
        
        # Potential uptake by bacteria (from PelBac.F90 'rum')
        potential_upt = f_B_n_P*fTB*bacteria_parameters.r_0B*b1c
        
        # correction of substrate quality depending on nutrient content (from PelBac.F90 'cuR1' & 'cuR6')
        f_r1_n_P = np.zeros(num_boxes)
        f_r6_n_P = np.zeros(num_boxes)
        for i in range(0,num_boxes):
            f_r1_n_P[i] = min(1.0, r1p_r1c[i]/bacteria_parameters.p_B_opt, r1n_r1c[i]/bacteria_parameters.n_B_opt)
            f_r6_n_P[i] = min(1.0, r6p_r6c[i]/bacteria_parameters.p_B_opt, r6n_r6c[i]/bacteria_parameters.n_B_opt)
    
    elif bacteria_parameters.bact_version==3:
        # Potential uptake by bacteria (from PelBac.F90 'rum')
        potential_upt = bacteria_parameters.r_0B*fTB*b1c

        # no correction of organic  material quality
        f_r1_n_P = np.ones(num_boxes)
        f_r6_n_P = np.ones(num_boxes)
        
    # Calculate the realized substrate uptake rate depending on the type of detritus and quality
    upt_r1c = (bacteria_parameters.v_B_r1*f_r1_n_P + bacteria_parameters.v_0B_r1*(1.0 - f_r1_n_P))*r1c # (from PelBac.F90 'ruR1c')
    upt_r2c = bacteria_parameters.v_B_r2*r2c   # (from PelBac.F90 'ruR2c')
    upt_r3c = bacteria_parameters.v_B_r3*r3c   # (from PelBac.F90 'ruR3c')
    upt_r6c = bacteria_parameters.v_B_r6*f_r6_n_P*r6c  # (from PelBac.F90 'ruR6c')
    realized_upt = constant_parameters.p_small + upt_r1c + upt_r2c + upt_r3c + upt_r6c    # (from PelBac.F90 'rut')
    
    # Actual uptake by bacteria (from PelBac.F90 'rug')
    actual_upt = np.zeros(num_boxes)
    for i in range(0,num_boxes):
        actual_upt[i] = min(potential_upt[i], realized_upt[i]) 
    
    # Carbon fluxes into bacteria
    dBcdt_upt_r1c = actual_upt*upt_r1c/realized_upt   # (from PelBac.F90 'ruR1c')
    dBcdt_upt_r2c = actual_upt*upt_r2c/realized_upt   # (from PelBac.F90 'ruR2c')
    dBcdt_upt_r3c = actual_upt*upt_r3c/realized_upt   # (from PelBac.F90 'ruR3c')
    dBcdt_upt_r6c = actual_upt*upt_r6c/realized_upt   # (from PelBac.F90 'ruR6c')
    
    # Organic Nitrogen and Phosphrous uptake
    dBcdt_upt_r1n = r1n_r1c*dBcdt_upt_r1c   # (from PelBac.F90 'ruR1n')
    dBcdt_upt_r6n = r6n_r6c*dBcdt_upt_r6c   # (from PelBac.F90 'ruR6n')
    dBcdt_upt_r1p = r1p_r1c*dBcdt_upt_r1c   # (from PelBac.F90 'ruR1p')
    dBcdt_upt_r6p = r6p_r6c*dBcdt_upt_r6c   # (from PelBac.F90 'ruR6p')
    
    # Bacteria respiration [mc C m^-3 s^-1] (from PelBac.F90 'rrc')
    dBcdt_rsp_o3c = (bacteria_parameters.gamma_B_a + bacteria_parameters.gamma_B_O*(1.0 - f_B_O))*actual_upt + bacteria_parameters.b_B*b1c*fTB

    # Fluxes from bacteria
    if bacteria_parameters.bact_version==1:
        
        # There is no Carbon excretion
        dBcdt_rel_r2c = np.zeros(num_boxes)
        dBcdt_rel_r3c = np.zeros(num_boxes)
        
        # Dissolved Nitrogen dynamics
        dBndt_upt_rel_n4n = (bn_bc - bacteria_parameters.n_B_opt)*b1c*bacteria_parameters.v_B_n    # (from PelBac.F90 'ren')
            
        # Dissolved Phosphorus dynamics
        dBpdt_upt_rel_n1p = (bp_bc - bacteria_parameters.p_B_opt)*b1c*bacteria_parameters.v_B_p    # (from PelBac.F90 'rep')

    elif bacteria_parameters.bact_version==2:
        print('This code does not support this parameterization option')
        
    elif bacteria_parameters.bact_version==3:
        # Carbon excretion as Semi-Labile (R2) and Semi-Refractory (R3) DOC
        dBcdt_rel_r2c = np.zeros(num_boxes)
        for i in range(0,num_boxes):
            dBcdt_rel_r2c[i] = max(1.-(bp_bc[i]/bacteria_parameters.p_B_opt) , (1.-(bn_bc[i]/bacteria_parameters.n_B_opt)*bacteria_parameters.v_B_c))
            dBcdt_rel_r2c[i] = max(0.,dBcdt_rel_r2c[i])*b1c[i]
        dBcdt_rel_r3c = actual_upt*(1.-bacteria_parameters.gamma_B_a)*(bacteria_parameters.gamma_B_a*bacteria_parameters.p_pu_ea_R3) # might need to toggle p_pu_ea_R3 between 0 and 0.015
        
        # Dissolved Nitrogen dynamics
        dBndt_upt_rel_n4n = (bn_bc - bacteria_parameters.n_B_opt)*b1c*bacteria_parameters.v_B_n    # (from PelBac.F90 'ren')

        # Dissolved Phosphorus dynamics
        dBpdt_upt_rel_n1p = (bp_bc - bacteria_parameters.p_B_opt)*b1c*bacteria_parameters.v_B_p    # (from PelBac.F90 'rep')
    
    # Term needed for denitrification flux (dn3ndt_denit) (from PelBac.F90 'flN6r')
    flPTn6r = (1.0 - f_B_O)*dBcdt_rsp_o3c*constant_parameters.omega_c*constant_parameters.omega_r

    return (dBcdt_lys_r1c, dBcdt_lys_r1n, dBcdt_lys_r1p, dBcdt_lys_r6c, dBcdt_lys_r6n, dBcdt_lys_r6p, 
            dBcdt_upt_r1c, dBcdt_upt_r6c, dBcdt_upt_r2c, dBcdt_upt_r3c,  dBpdt_upt_rel_n1p, dBndt_upt_rel_n4n,
            dBcdt_rel_r2c, dBcdt_rel_r3c, dBcdt_rsp_o3c, flPTn6r, f_B_O, f_B_n, f_B_p)
    

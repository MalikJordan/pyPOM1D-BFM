import numpy as np
import sys
from bfm.bfm50.Functions.other_functions import insw_vector, eTq_vector, get_concentration_ratio
from bfm.parameters import Phyto1, Phyto2, Phyto3, Phyto4
from pom.parameters import PomBfm

phyto1_prameters = Phyto1()
phyto2_prameters = Phyto2()
phyto3_prameters = Phyto3()
phyto4_prameters = Phyto4()

pom_bfm_parameters = PomBfm()

def phyto_eqns(d3state, bfm_phys_vars, phyto_parameters, env_parameters, constant_parameters, del_z, group, irradiance, pc, pn, pp, pl, suspended_sediments, temp, time, xEPS):
    """ Calculates the terms needed for the phytoplnaktion biological rate equations
        Equations come from the BFM user manual
    """

    num_boxes = d3state.shape[0]

    # Species concentrations
    n1p = d3state[:,1]               # Phosphate (mmol P m^-3)
    n3n = d3state[:,2]               # Nitrate (mmol N m^-3)
    n4n = d3state[:,3]               # Ammonium (mmol N m^-3)
    n5s = d3state[:,5]               # Silicate (mmol Si m^-3)
    p1l = d3state[:,13]              # Diatoms chlorophyll (mg Chl-a m^-3)
    p1s = d3state[:,14]              # Diatoms silicate (mmol Si m^-3) 
    p2l = d3state[:,18]              # NanoFlagellates chlorophyll (mg Chl-a m^-3)
    p3l = d3state[:,22]              # Picophytoplankton chlorophyll (mg Chl-a m^-3)
    p4l = d3state[:,26]              # Large phytoplankton chlorophyll (mg Chl-a m^-3)
    r6c = d3state[:,44]              # Particulate organic carbon (mg C m^-3)
    
    # Concentration ratios (constituents quota in phytoplankton)
    pn_pc = get_concentration_ratio(pn, pc, constant_parameters.p_small)
    pp_pc = get_concentration_ratio(pp, pc, constant_parameters.p_small)
    pl_pc = get_concentration_ratio(pl, pc, constant_parameters.p_small)
    
    #--------------------------------------------------------------------------
    # Temperature response of Phytoplankton Include cut-off at low temperature if p_temp>0
    et = eTq_vector(temp, env_parameters.basetemp, env_parameters.q10z)
    fTP = np.zeros(num_boxes)
    for i in range(0,num_boxes):
        fTP[i] = max(0.0, et[i] - phyto_parameters.p_temp)

    #--------------------------------------------------------------------------
    # Nutrient limitations (intracellular and extracellular) fpplim is the 
    # combined non-dimensional factor limiting photosynthesis
    # from Phyto.F90 lines 268-308
    in1p = np.zeros(num_boxes)
    in1n = np.zeros(num_boxes)
    for i in range(0,num_boxes):
        in1p[i] = min(1.0, max(constant_parameters.p_small, (pp_pc[i] - phyto_parameters.phi_Pmin)/(phyto_parameters.phi_Popt - phyto_parameters.phi_Pmin))) # (from Phyto.F90 'iN1p')
        in1n[i] = min(1.0, max(constant_parameters.p_small, (pn_pc[i] - phyto_parameters.phi_Nmin)/(phyto_parameters.phi_Nopt - phyto_parameters.phi_Nmin)))  # (from Phyto.F90 'iNIn')
    
    if group == 1:
        fpplim = np.zeros(num_boxes)
        for i in range(num_boxes):
            fpplim[i] = min(1.0, n5s[i]/(n5s[i] + phyto_parameters.h_Ps + (phyto_parameters.rho_Ps*p1s[i])))
    else:
        fpplim = np.ones(num_boxes)

    #--------------------------------------------------------------------------
    # multiple nutrient limitation, Liebig rule (from Phyto.F90 line ~318, iN)
    multiple_nut_lim = np.zeros(num_boxes)
    for i in range(0,num_boxes):
        multiple_nut_lim[i] = min(in1p[i], in1n[i])
    
    # tN only controls sedimentation of phytoplanton (liebig)
    tN = np.zeros(num_boxes)
    for i in range(0,num_boxes):
        tN[i] = min(multiple_nut_lim[i],fpplim[i])
    
    r = np.zeros(num_boxes)
    irr = np.zeros(num_boxes)
    exponent = np.zeros(num_boxes)
    light_lim = np.zeros(num_boxes)

    for i in range(0,num_boxes):
        r[i] = xEPS[i] * del_z[i]
        r[i] = irradiance[i]/xEPS[i]/del_z[i]*(1.0 - np.exp(-r[i]))        
        irr[i] = max(constant_parameters.p_small, r[i] * pom_bfm_parameters.sec_per_day)

        exponent[i] = pl_pc[i]*phyto_parameters.alpha_chl/phyto_parameters.rP0*irr[i]     # Compute exponent E_PAR/E_K = alpha0/PBmax (part of eqn. 2.2.4)

        light_lim[i] = (1.0 - np.exp(-exponent[i]))     # light limitation factor (from Phyto.f90 line 374, eiPPY)


    #--------------------------------------------------------------------------
    # total photosynthesis (from Phyto.F90 line ~380, sum)
    photosynthesis = phyto_parameters.rP0*fTP*light_lim*fpplim

    #--------------------------------------------------------------------------
    # Lysis nad excretion
    # nutr. -stress lysis (from Phyto.F90 lines ~385-387, sdo)
    nut_stress_lysis = (phyto_parameters.h_Pnp/(multiple_nut_lim + phyto_parameters.h_Pnp))*phyto_parameters.d_P0
    nut_stress_lysis += phyto_parameters.p_seo*pc/(pc + phyto_parameters.p_sheo + constant_parameters.p_small)

    # activity excretion (Phyto.F90 line 389)
    activity_excretion = photosynthesis*phyto_parameters.betaP

    # nutrient stress excretion from Phyto.F90 line 396
    nut_stress_excretion = photosynthesis*(1.0 - phyto_parameters.betaP)*(1.0 - multiple_nut_lim)

    #--------------------------------------------------------------------------
    # Apportioning over R1 and R6: Cell lysis generates both DOM and POM
    pe_R6 = np.zeros(num_boxes)
    for i in range(0,num_boxes):
        pe_R6[i] = min(phyto_parameters.phi_Pmin/(pp_pc[i] + constant_parameters.p_small), phyto_parameters.phi_Nmin/(pn_pc[i] + constant_parameters.p_small))
        pe_R6[i] = min(1.0, pe_R6[i])
    rr6c = pe_R6*nut_stress_lysis*pc
    rr1c = (1.0 - pe_R6)*nut_stress_lysis*pc

    #--------------------------------------------------------------------------
    # Respiration rate
    # activity (from Phyto.F90 line 416)
    activity_rsp = phyto_parameters.gammaP*(photosynthesis - activity_excretion - nut_stress_excretion)

    # basal (from Phyto.F90 line 417)
    basal_rsp = fTP*phyto_parameters.bP

    # total (from Phyto.F90 line 418)
    total_rsp = activity_rsp + basal_rsp

    # total actual respiration
    dPcdt_rsp_o3c = total_rsp*pc

    #--------------------------------------------------------------------------
    # Production, productivity and C flows
    # Phytoplankton gross primary production [mg C m^-3 s^-1]
    dPcdt_gpp_o3c = photosynthesis*pc

    # specific loss terms (from Phyto.F90 line 428)
    specific_loss_terms = activity_excretion + nut_stress_excretion + total_rsp + nut_stress_lysis

    # All activity excretions are assigned to R1
    # p_switchDOC=1 and P_netgrowth=FLASE: [mg C m^-3 s^-1]
    rr1c += activity_excretion*pc + nut_stress_excretion*pc
    dPcdt_exu_r2c = np.zeros(num_boxes)

    # Phytoplankton DOM cell lysis- carbon lost to DOM [mg C m^-3 s^-1]
    dPcdt_lys_r1c = rr1c

    # Phytoplankton POM cell lysis- carbon lost to POM (eqn. 2.2.9) [mg C m^-3 s^-1]
    dPcdt_lys_r6c = rr6c

    #--------------------------------------------------------------------------
    # Potential-Net primary production
    # from Phyto.F90 line 455
    sadap = fTP*phyto_parameters.rP0
    
    # Net production (from Phyto.F90 line 457, 'run')
    net_production = np.zeros(num_boxes)
    for i in range(0,num_boxes):
        net_production[i] = max(0.0, (photosynthesis[i] - specific_loss_terms[i])*pc[i])
    
    #--------------------------------------------------------------------------
    # Nutrient Uptake: calculate maximum uptake of N, P based on affinity

    cqun3 = phyto_parameters.h_Pn/(constant_parameters.p_small + phyto_parameters.h_Pn + n4n)

    # max potential uptake of N3 (from Phyto.F90 'rumn3')
    max_upt_n3n = phyto_parameters.a_N*n3n*pc*cqun3

    # max potential uptake of N4 (from Phyto.F90 'rumn4')
    max_upt_n4n = phyto_parameters.a_N*n4n*pc

    # max potential uptake of DIN (from Phyto.F90 'rumn')
    max_upt_DIN = max_upt_n3n + max_upt_n4n

    # max pot. uptake of PO4 (from Phyto.F90 line 468)
    rump = phyto_parameters.a_P*n1p*pc

    #--------------------------------------------------------------------------
    # Nutrient dynamics: NITROGEN

    # Intracellular missing amount of N (from Phyto.F90)
    misn = sadap*(phyto_parameters.p_xqn*phyto_parameters.phi_Nmax*pc - pn)

    # N uptake based on net assimilat. C (from Phyto.F90)
    rupn = phyto_parameters.p_xqn*phyto_parameters.phi_Nmax*net_production

    # actual uptake of NI (from Phyto.F90, 'runn')
    dPndt_upt = np.zeros(num_boxes)
    for i in range(0,num_boxes):
        dPndt_upt[i] = min(max_upt_DIN[i], rupn[i] + misn[i])

    # if nitrogen uptake rate is positive, then uptake is divided between coming from the nitrate and ammonium reservoir
    # if nitrogen uptake is negative, all nitrogen goes to the DOM pool
    upt_switch_n = insw_vector(dPndt_upt)

    # actual uptake of n3n (from Phyto.F90, 'runn3')
    dPndt_upt_n3n = upt_switch_n*dPndt_upt*max_upt_n3n/(constant_parameters.p_small + max_upt_DIN)

    # actual uptake of n4n (from Phyto.F90, 'runn4')
    dPndt_upt_n4n = upt_switch_n*dPndt_upt*max_upt_n4n/(constant_parameters.p_small + max_upt_DIN)

    extra_n = -dPndt_upt*(1.0 - upt_switch_n)

    #--------------------------------------------------------------------------
    # Nutrient dynamics: PHOSPHORUS

    # intracellular missing amount of P (from Phyto.F90 line 514)
    misp = sadap*(phyto_parameters.p_xqp*phyto_parameters.phi_Pmax*pc-pp)

    # P uptake based on C uptake (from Phyto.F90 line 517)
    rupp = phyto_parameters.p_xqp*phyto_parameters.phi_Pmax*net_production

    # Actual uptake
    runp = np.zeros(num_boxes)
    for i in range(0,num_boxes):
        runp[i] = min(rump[i], rupp[i] + misp[i])
    upt_switch_p = insw_vector(runp)
    dPpdt_upt_n1p = runp*upt_switch_p

    # is uptake is negative flux goes to DIP (r1p) pool
    dPpdt_upt_r1p = -runp*(1.0 - upt_switch_p)

    #--------------------------------------------------------------------------
    # Excretion of N and P to PON and POP
    dPndt_lys_r6n = pe_R6*nut_stress_lysis*pn
    dPndt_lys_r1n = nut_stress_lysis*pn - dPndt_lys_r6n

    dPpdt_lys_r6p = pe_R6*nut_stress_lysis*pp
    dPpdt_lys_r1p = nut_stress_lysis*pp - dPpdt_lys_r6p

    #--------------------------------------------------------------------------
    # Nutrient dynamics: SILICATE
    if group == 1:
        dPsdt_upt_n5s = np.zeros(num_boxes)
        for i in range(0,num_boxes):
            # Gross uptake of silicate excluding respiratory costs (from Phyto.F90, 'runs')
            dPsdt_upt_n5s[i] = max(0.0, phyto_parameters.phi_Sopt*pc[i]*(photosynthesis[i] - basal_rsp[i]))
        # losses of Si (from Phyto.F90)
        dPsdt_lys_r6s = nut_stress_lysis*p1s
    else:
        dPsdt_upt_n5s = np.zeros(num_boxes)
        dPsdt_lys_r6s = nut_stress_lysis*p1s

    #--------------------------------------------------------------------------
    # Chl-a synthesis and photoacclimation
    if phyto_parameters.chl_switch == 1:
        # dynamical chl:c ratio from Fortran code Phyto.F90
        rho_chl = np.zeros(num_boxes)
        for i in range(0,num_boxes):
            rho_chl[i] = phyto_parameters.theta_chl0*min(1.0, phyto_parameters.rP0*light_lim[i]*pc[i]/(phyto_parameters.alpha_chl*(pl[i] + constant_parameters.p_small)*irr[i]))
        
        # Chlorophyll synthesis from Fortran code Phyto.F90, rate_chl
        dPldt_syn = rho_chl*(photosynthesis - nut_stress_excretion - activity_excretion - activity_rsp)*pc - nut_stress_lysis*pl
    elif phyto_parameters.chl_switch == 3:
        rho_chl = np.zeros(num_boxes)
        chl_opt = np.zeros(num_boxes)
        dPldt_syn = np.zeros(num_boxes)
        # chl_relax = 0.
        for i in range(0,num_boxes):
            rho_chl[i] = phyto_parameters.theta_chl0*min(1.0,(photosynthesis[i] - nut_stress_excretion[i] - activity_excretion[i] - activity_rsp[i])*pc[i]/(phyto_parameters.alpha_chl*(pl[i] + constant_parameters.p_small)*irr[i]))
            chl_opt[i] = phyto_parameters.p_EpEk_or*phyto_parameters.rP0*pc[i]/(phyto_parameters.alpha_chl*irr[i]+constant_parameters.p_small)
            # for Phyto2 : chl_opt = 0 & chl_relax = 0 (from Pelagic_Ecology.nml)
            dPldt_syn[i] = rho_chl[i]*(photosynthesis[i] - nut_stress_excretion[i] - activity_excretion[i] - activity_rsp[i])*pc[i] - (nut_stress_lysis[i] + basal_rsp[i])*pl[i] - max(0.0,pl[i]-chl_opt[i])*phyto_parameters.p_tochl_relt
    else:
        sys.exit("Warning: This code does not support other chl systhesis parameterizations")
    
    #--------------------------------------------------------------------------
    # Sedimentation
    # sedi = bfm_phys_vars.phyto_sedimentation[:,group-1]
    bfm_phys_vars.phyto_sedimentation[:,group-1] = phyto_parameters.p_rPIm*np.ones(num_boxes) # from PelGlobal.F90
    if (phyto_parameters.p_res > 0): # from Phyto.F90 
        for i in range(0,num_boxes):
            bfm_phys_vars.phyto_sedimentation[i,group-1] = bfm_phys_vars.phyto_sedimentation[i,group-1] + phyto_parameters.p_res*max(0., phyto_parameters.p_esNI-tN[i])
    
    #--------------------------------------------------------------------------
    return (dPcdt_gpp_o3c, dPcdt_rsp_o3c, dPcdt_lys_r1c, dPcdt_lys_r6c, dPcdt_exu_r2c, 
            dPndt_upt_n3n, dPndt_upt_n4n, extra_n, dPndt_lys_r1n, dPndt_lys_r6n, 
            dPpdt_upt_n1p, dPpdt_upt_r1p, dPpdt_lys_r1p, dPpdt_lys_r6p, 
            dPldt_syn, dPsdt_upt_n5s, dPsdt_lys_r6s, bfm_phys_vars)


def chlorophylla(d3state):
    """
    Calculates the sum of all phytoplankton chlorophyll constituents
    """
    chl = d3state[:,13] + d3state[:,18] + d3state[:,22] + d3state[:,26]

    return chl

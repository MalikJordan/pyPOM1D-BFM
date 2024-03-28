import numpy as np
import os
current_path = os.getcwd()


class Bfm:
    def __init__(self):
        self.pom_bfm = True
        self.AssignAirPelFluxesInBFMFlag = True
        
        # Number of state variables in bfm model
        self.num_d3_box_states = 50

        # Initial values for pelagic state variables
        self.o2o0 = 219.0
        self.n1p0 = 0.003
        self.n3n0 = 0.04
        self.n4n0 = 0.008
        self.n5s0 = 0.0
        self.n6r0 = 0.0
        self.o3c0 = 0.0
        self.o3h0 = 0.0
        self.o4n0 = 0.0
        self.p1c0 = 0.0
        self.p2c0 = 11.5
        self.p3c0 = 0.0
        self.p4c0 = 0.0
        self.z3c0 = 0.0
        self.z4c0 = 0.0
        self.z5c0 = 11.5
        self.z6c0 = 0.0
        self.b1c0 = 5.0
        self.r1c0 = 12.4
        self.r2c0 = 0.0
        self.r3c0 = 0.0
        self.r6c0 = 12.4


class BfmInputFiles:
    def __init__(self):
        self.phytoc = current_path + '/inputs/BFM17_BERM_INIT/init_prof_Pc_150m_bermuda_killworth.da'
        self.zooc = current_path + '/inputs/BFM17_BERM_INIT/init_prof_Zc_150m_bermuda_killworth.da'
        self.poc = current_path + '/inputs/BFM17_BERM_INIT/init_prof_POC_150m_bermuda_killworth.da'
        self.doc = current_path + '/inputs/BFM17_BERM_INIT/init_prof_DOC_150m_bermuda_killworth.da'
        self.phos = current_path + '/inputs/BFM17_BERM_INIT/init_prof_P_150m_bermuda_killworth.da'
        self.nit = current_path + '/inputs/BFM17_BERM_INIT/init_prof_N_150m_bermuda_killworth.da'
        self.am = current_path + '/inputs/BFM17_BERM_INIT/init_prof_Am_150m_bermuda_killworth.da'
        self.oxy = current_path + '/inputs/BFM17_BERM_INIT/init_prof_Oxy_150m_bermuda_killworth.da'


class Bacteria:
    def __init__(self):
        self.a_1B = 0.0
        self.a_1B = 0.0
        self.a_4B = 0.0
        self.b_B = 0.01
        self.bact_version = 3
        self.beta_B = 0.0
        self.d_0B = 0.1
        self.d_B_d = 0.0
        self.gamma_B_O = 0.2
        self.gamma_B_a = 0.6
        self.h_B_O = 30.0
        self.h_B_n = 5.0
        self.h_B_p = 1.0
        self.n_B_min = 0.0167
        self.n_B_opt = 0.0167
        self.p_B_min = 0.00185
        self.p_B_opt = 0.00185
        self.p_pu_ea_R3 = 0
        self.r_0B = 8.38
        self.v_0B_r1 = 0.0
        self.v_B_c = 1.0
        self.v_B_n = 1.0
        self.v_B_p = 1.0
        self.v_B_r1 = 0.3
        self.v_B_r2 = 0.0
        self.v_B_r3 = 0.0
        self.v_B_r6 = 0.01


class Co2Flux:
    def __init__(self):
        self.atm_co2_0 = 370.0
        self.c1 = 2073.1
        self.c2 = 125.62
        self.c3 = 3.6276
        self.c4 = 0.043219
        self.d = 0.31
        self.m2maxit = 100.0
        self.m2phdelt = 0.3
        self.m2xacc = 1e-10
        self.ph_initial = 8.12
        self.schmidt_o3c = 660.0


class Constants:
    def __init__(self):
        self.c_to_kelvin = 273.15
        self.cm2m = 0.01
        self.e2w = 0.217
        self.epsilon_c = 0.6
        self.epsilon_n = 0.72
        self.epsilon_p = 0.832
        self.g_per_mg = 0.001
        self.hours_per_day = 24.0
        self.mol_per_mmol = 0.001
        self.omega_c = 0.08333333333333333
        self.omega_dn = 1.25
        self.omega_n = 2.0
        self.omega_nr = 0.625
        self.omega_r = 0.5
        self.p_small = 1e-20
        self.p_pe_R1c = 0.60
        self.sec_per_day = 86400.0
        self.uatm_per_atm = 1000000.0
        self.umol_per_mol = 1000000.0


class Environment:
    def __init__(self):
        self.basetemp = 20.0
        self.c_r6 = 0.0001
        self.del_z = 5.0
        self.epsilon_PAR = 0.4
        self.p_eps0 = 0.0435
        self.p_epsESS = 0.0
        self.p_epsR6 = 0.0001
        self.q10b = 2.95
        self.q10n = 2.0
        self.q10n5 = 1.49
        self.q10p = 2.0
        self.q10z = 2.0
        self.w_win = 20.0
        self.w_sum = 10.0
        self.t_win = 8.0
        self.t_sum = 28.0
        self.tde = 1.0
        self.s_win = 37.0
        self.s_sum = 34.0
        self.qs_win = 20.0
        self.qs_sum = 300.0


class MesoZoo3:
    def __init__(self):
        self.bZ = 0.01
        self.betaZ = 0.3
        self.d_Z = 0.02
        self.d_Zdns = 0.01
        self.etaZ = 0.6
        self.gammaZ = 2.0
        self.n_Zopt = 0.015
        self.nu_z = 2.5e-03
        self.p_qncMEZ = 0.01258
        self.p_qpcMEZ = 7.862e-04
        self.p_Zopt = 0.00167
        self.r_Z0 = 2.0
        self.z_o2o = 30.0


class MesoZoo4:
    def __init__(self):
        self.bZ = 0.02
        self.betaZ = 0.35
        self.d_Z = 0.02
        self.d_Zdns = 0.01
        self.etaZ = 0.6
        self.gammaZ = 2.0
        self.n_Zopt = 0.015
        self.nu_z = 0.025
        self.p_qncMEZ = 0.01258
        self.p_qpcMEZ = 7.862e-04
        self.p_Zopt = 0.00167
        self.r_Z0 = 2.0
        self.z_o2o = 30.0


class MicroZoo5:
    def __init__(self):
        self.bZ = 0.02
        self.betaZ = 0.25
        self.d_Z = 1.0e-06
        self.d_ZO = 0.25
        self.etaZ = 0.5
        self.h_Z_F = 200.0
        self.mu_z = 50.0
        self.n_Zopt = 0.0128
        self.p_Zopt = 0.00185
        self.r_Z0 = 2.0
        self.z_o2o = 0.5


class MicroZoo6:
    def __init__(self):
        self.bZ = 0.02
        self.betaZ = 0.35
        self.d_Z = 1.0e-06
        self.d_ZO = 0.25
        self.etaZ = 0.3
        self.h_Z_F = 200.0
        self.mu_z = 50.0
        self.n_Zopt = 0.0128
        self.p_Zopt = 0.00185
        self.r_Z0 = 5.0
        self.z_o2o = 0.5


class OxygenReaeration:
    def __init__(self):
        self.d = 0.31
        self.k1 = 1953.4
        self.k2 = 128.0
        self.k3 = 3.9918
        self.k4 = 0.050091
        self.schmidt_o2o = 660.0


class PelChem:
    def __init__(self):
        self.calc_alkalinity = False
        self.calc_bacteria = True
        self.h_o = 10.0
        self.h_r = 1.0
        self.lambda_N3denit = 0.35
        self.lambda_n4nit = 0.01
        self.lambda_n6reox = 0.05
        self.lambda_srmn = 0.02
        self.m_o = 1.0
        self.p_rR6m = 1.0
        

class Phyto1:
    def __init__(self):
        self.a_Fe = 2.3148148148148148e-09
        self.a_N = 0.025
        self.a_P = 2.5e-03
        self.a_S = 0.0
        self.alpha_chl = 1.1e-05
        self.bP = 0.05
        self.betaP = 0.01
        self.c_P = 0.03
        self.chl_switch = 3
        self.d_P0 = 0.01
        self.gammaP = 0.1
        self.h_Pn = 1.0
        self.h_Pnp = 0.1
        self.h_Ps = 1.0
        self.p_burvel_PI = 0
        self.p_esNI = 0.70
        self.p_EpEk_or = 3.0
        self.p_res = 5.0
        self.p_rPIm = 0.0
        self.p_seo = 0.0
        self.p_sheo = 0.0
        self.p_temp = 0.0
        self.p_xqn = 2.0
        self.p_xqp = 2.0
        self.p_tochl_relt = 0.25
        self.phi_Nmax = 0.0126
        self.phi_Nmin = 6.87e-03
        self.phi_Nopt = 0.0126
        self.phi_Pmax = 0.001572
        self.phi_Pmin = 4.29e-04
        self.phi_Popt = 0.000786
        self.phi_Smin = 0.0
        self.phi_Sopt = 0.01
        self.rP0 = 2.5
        self.rho_Ps = 0.0
        self.si_switch = 1
        self.theta_chl0 = 0.035
        
        
class Phyto2:
    def __init__(self):
        self.a_Fe = 2.3148148148148148e-09
        self.a_N = 0.025
        self.a_P = 2.5e-03
        self.a_S = 0.0
        self.alpha_chl = 1.52e-05
        self.bP = 0.05
        self.betaP = 0.05
        self.c_P = 0.03
        self.chl_switch = 3
        self.d_P0 = 0.05
        self.gammaP = 0.05
        self.h_Pn = 1.5
        self.h_Pnp = 0.1
        self.h_Ps = 0.0
        self.p_burvel_PI = 0
        self.p_esNI = 0.75
        self.p_EpEk_or = 0.0
        self.p_res = 0.5
        self.p_rPIm = 0.0
        self.p_seo = 0.0
        self.p_sheo = 0.0
        self.p_temp = 0.0
        self.p_xqn = 1.0
        self.p_xqp = 1.0
        self.p_tochl_relt = 0.0
        self.phi_Nmax = 0.0126
        self.phi_Nmin = 6.87e-03
        self.phi_Nopt = 0.0126
        self.phi_Pmax = 7.86e-04
        self.phi_Pmin = 4.29e-04
        self.phi_Popt = 0.000786
        self.phi_Smin = 0.0
        self.phi_Sopt = 0.0
        self.rP0 = 1.6
        self.rho_Ps = 0.0
        self.si_switch = 0
        self.theta_chl0 = 0.016
        
        
class Phyto3:
    def __init__(self):
        self.a_Fe = 2.3148148148148148e-09
        self.a_N = 0.025
        self.a_P = 2.5e-03
        self.a_S = 0.0
        self.alpha_chl = 0.7e-05
        self.bP = 0.05
        self.betaP = 0.1
        self.c_P = 0.03
        self.chl_switch = 3
        self.d_P0 = 0.01
        self.gammaP = 0.2
        self.h_Pn = 0.1
        self.h_Pnp = 0.1
        self.h_Ps = 0.0
        self.p_burvel_PI = 0
        self.p_esNI = 0.75
        self.p_EpEk_or = 3.0
        self.p_res = 0.5
        self.p_rPIm = 0.0
        self.p_seo = 0.0
        self.p_sheo = 0.0
        self.p_temp = 0.0
        self.p_xqn = 2.0
        self.p_xqp = 2.0
        self.p_tochl_relt = 0.25
        self.phi_Nmax = 0.0126
        self.phi_Nmin = 6.87e-03
        self.phi_Nopt = 0.0126
        self.phi_Pmax = 0.001572
        self.phi_Pmin = 4.29e-04
        self.phi_Popt = 0.000786
        self.phi_Smin = 0.0
        self.phi_Sopt = 0.0
        self.rP0 = 1.02
        self.rho_Ps = 0.0
        self.si_switch = 0
        self.theta_chl0 = 0.02
        
        
class Phyto4:
    def __init__(self):
        self.a_Fe = 2.3148148148148148e-09
        self.a_N = 0.025
        self.a_P = 2.5e-03
        self.a_S = 0.0
        self.alpha_chl = 6.8e-06
        self.bP = 0.1
        self.betaP = 0.15
        self.c_P = 0.03
        self.chl_switch = 3
        self.d_P0 = 0.2
        self.gammaP = 0.1
        self.h_Pn = 1.0
        self.h_Pnp = 0.5
        self.h_Ps = 0.0
        self.p_burvel_PI = 0
        self.p_esNI = 0.75
        self.p_EpEk_or = 3.0
        self.p_res = 5.0
        self.p_rPIm = 0.0
        self.p_seo = 0.5
        self.p_sheo = 100.0
        self.p_temp = 0.0
        self.p_xqn = 2.0
        self.p_xqp = 2.0
        self.p_tochl_relt = 0.25
        self.phi_Nmax = 0.0126
        self.phi_Nmin = 6.87e-03
        self.phi_Nopt = 0.0126
        self.phi_Pmax = 0.001572
        self.phi_Pmin = 4.29e-04
        self.phi_Popt = 0.000786
        self.phi_Smin = 0.0
        self.phi_Sopt = 0.0
        self.rP0 = 0.83
        self.rho_Ps = 0.0
        self.si_switch = 0
        self.theta_chl0 = 0.035
        
        
class ZooAvailability:
    def __init__(self):
        self.del_z3p1 = 0.0
        self.del_z3p2 = 0.0
        self.del_z3p3 = 0.0
        self.del_z3p4 = 0.0
        self.del_z3z3 = 1.0
        self.del_z3z4 = 1.0
        self.del_z3z5 = 0.0
        self.del_z3z6 = 0.0
        self.del_z4p1 = 1.0
        self.del_z4p2 = 0.1
        self.del_z4p3 = 0.0
        self.del_z4p4 = 0.7
        self.del_z4z3 = 0.0
        self.del_z4z4 = 1.0
        self.del_z4z5 = 1.0
        self.del_z4z6 = 0.0
        self.del_z5b1 = 0.1
        self.del_z5p1 = 0.5
        self.del_z5p2 = 1.0
        self.del_z5p3 = 0.5
        self.del_z5p4 = 0.0
        self.del_z5z5 = 1.0
        self.del_z5z6 = 0.8
        self.del_z6b1 = 1.0
        self.del_z6p1 = 0.0
        self.del_z6p2 = 0.1
        self.del_z6p3 = 1.0
        self.del_z6p4 = 0.0
        self.del_z6z5 = 0.0
        self.del_z6z6 = 0.2

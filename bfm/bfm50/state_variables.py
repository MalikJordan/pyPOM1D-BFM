import numpy as np


def bfm_rates(d3state, species, sec_per_day,
              do2o_dt, dn1p_dt, dn3n_dt, dn4n_dt, do4n_dt, dn5s_dt, dn6r_dt, db1c_dt, db1n_dt, db1p_dt, 
              dp1c_dt, dp1n_dt, dp1p_dt, dp1l_dt, dp1s_dt, dp2c_dt, dp2n_dt, dp2p_dt, dp2l_dt, 
              dp3c_dt, dp3n_dt, dp3p_dt, dp3l_dt, dp4c_dt, dp4n_dt, dp4p_dt, dp4l_dt, dz3c_dt, dz3n_dt, dz3p_dt,
              dz4c_dt, dz4n_dt, dz4p_dt, dz5c_dt, dz5n_dt, dz5p_dt, dz6c_dt, dz6n_dt, dz6p_dt, dr1c_dt, dr1n_dt, dr1p_dt, 
              dr2c_dt, dr3c_dt, dr6c_dt, dr6n_dt, dr6p_dt, dr6s_dt, do3c_dt, do3h_dt):

    rates = np.zeros_like(d3state)

    if 'O2o' in species:  rates[:,species['O2o']] = do2o_dt     # Dissolved oxygen (mg O_2 m^-3)
    
    if 'N1p' in species:  rates[:,species['N1p']] = dn1p_dt     # Phosphate (mmol P m^-3)
    
    if 'N3n' in species:  rates[:,species['N3n']] = dn3n_dt     # Nitrate (mmol N m^-3)
    
    if 'N4n' in species:  rates[:,species['N4n']] = dn4n_dt     # Ammonium (mmol N m^-3)
    
    if 'O4n' in species:  rates[:,species['O4n']] = do4n_dt     # Nitrogen sink (mmol N m^-3)

    if 'N5s' in species:  rates[:,species['N5s']] = dn5s_dt     # Silicate (mmol Si m^-3)

    if 'N6r' in species:  rates[:,species['N6r']] = dn6r_dt     # Reduction equivalents (mmol S m^-3)

    if 'B1c' in species:  rates[:,species['B1c']] = db1c_dt     # Pelagic bacteria carbon (mg C m^-3)

    if 'B1n' in species:  rates[:,species['B1n']] = db1n_dt     # Pelagic bacteria nitrogen (mmol N m^-3)

    if 'B1p' in species:  rates[:,species['B1p']] = db1p_dt     # Pelagic bacteria phosphate (mmol P m^-3)

    if 'P1c' in species:  rates[:,species['P1c']] = dp1c_dt     # Diatoms carbon (mg C m^-3)

    if 'P1n' in species:  rates[:,species['P1n']] = dp1n_dt     # Diatoms nitrogen (mmol N m^-3)

    if 'P1p' in species:  rates[:,species['P1p']] = dp1p_dt     # Diatoms phosphate (mmol P m^-3)

    if 'P1l' in species:  rates[:,species['P1l']] = dp1l_dt     # Diatoms chlorophyll (mg Chl-a m^-3)

    if 'P1s' in species:  rates[:,species['P1s']] = dp1s_dt     # Diatoms silicate (mmol Si m^-3) 

    if 'P2c' in species:  rates[:,species['P2c']] = dp2c_dt     # NanoFlagellates carbon (mg C m^-3)

    if 'P2n' in species:  rates[:,species['P2n']] = dp2n_dt     # NanoFlagellates nitrogen (mmol N m^-3)

    if 'P2p' in species:  rates[:,species['P2p']] = dp2p_dt     # NanoFlagellates phosphate (mmol P m^-3)

    if 'P2l' in species:  rates[:,species['P2l']] = dp2l_dt     # NanoFlagellates chlorophyll (mg Chl-a m^-3)

    if 'P3c' in species:  rates[:,species['P3c']] = dp3c_dt     # Picophytoplankton carbon (mg C m^-3)

    if 'P3n' in species:  rates[:,species['P3n']] = dp3n_dt     # Picophytoplankton nitrogen (mmol N m^-3)

    if 'P3p' in species:  rates[:,species['P3p']] = dp3p_dt     # Picophytoplankton phosphate (mmol P m^-3)

    if 'P3l' in species:  rates[:,species['P3l']] = dp3l_dt     # Picophytoplankton chlorophyll (mg Chl-a m^-3)

    if 'P4c' in species:  rates[:,species['P4c']] = dp4c_dt     # Large phytoplankton carbon (mg C m^-3)

    if 'P4n' in species:  rates[:,species['P4n']] = dp4n_dt     # Large phytoplankton nitrogen (mmol N m^-3)

    if 'P4p' in species:  rates[:,species['P4p']] = dp4p_dt     # Large phytoplankton phosphate (mmol P m^-3) 

    if 'P4l' in species:  rates[:,species['P4l']] = dp4l_dt     # Large phytoplankton chlorophyll (mg Chl-a m^-3)

    if 'Z3c' in species:  rates[:,species['Z3c']] = dz3c_dt     # Carnivorous mesozooplankton carbon (mg C m^-3)

    if 'Z3n' in species:  rates[:,species['Z3n']] = dz3n_dt     # Carnivorous mesozooplankton nitrogen (mmol N m^-3)

    if 'Z3p' in species:  rates[:,species['Z3p']] = dz3p_dt     # Carnivorous mesozooplankton phosphate (mmol P m^-3)

    if 'Z4c' in species:  rates[:,species['Z4c']] = dz4c_dt     # Omnivorous mesozooplankton carbon (mg C m^-3)

    if 'Z4n' in species:  rates[:,species['Z4n']] = dz4n_dt     # Omnivorous mesozooplankton nitrogen (mmol N m^-3)

    if 'Z4p' in species:  rates[:,species['Z4p']] = dz4p_dt     # Omnivorous mesozooplankton phosphate (mmol P m^-3)

    if 'Z5c' in species:  rates[:,species['Z5c']] = dz5c_dt     # Microzooplankton carbon (mg C m^-3)

    if 'Z5n' in species:  rates[:,species['Z5n']] = dz5n_dt     # Microzooplankton nitrogen (mmol N m^-3)

    if 'Z5p' in species:  rates[:,species['Z5p']] = dz5p_dt     # Microzooplankton phosphate (mmol P m^-3)

    if 'Z6c' in species:  rates[:,species['Z6c']] = dz6c_dt     # Heterotrophic flagellates carbon (mg C m^-3)

    if 'Z6n' in species:  rates[:,species['Z6n']] = dz6n_dt     # Heterotrophic flagellates nitrogen (mmol N m^-3)

    if 'Z6p' in species:  rates[:,species['Z6p']] = dz6p_dt     # Heterotrophic flagellates phosphate (mmol P m^-3)

    if 'R1c' in species:  rates[:,species['R1c']] = dr1c_dt     # Labile dissolved organic carbon (mg C m^-3)

    if 'R1n' in species:  rates[:,species['R1n']] = dr1n_dt     # Labile dissolved organic nitrogen (mmol N m^-3)

    if 'R1p' in species:  rates[:,species['R1p']] = dr1p_dt     # Labile dissolved organic phosphate (mmol P m^-3)

    if 'R2c' in species:  rates[:,species['R2c']] = dr2c_dt     # Semi-labile dissolved organic carbon (mg C m^-3)

    if 'R3c' in species:  rates[:,species['R3c']] = dr3c_dt     # Semi-refractory Dissolved Organic Carbon (mg C m^-3)

    if 'R6c' in species:  rates[:,species['R6c']] = dr6c_dt     # Particulate organic carbon (mg C m^-3)

    if 'R6n' in species:  rates[:,species['R6n']] = dr6n_dt     # Particulate organic nitrogen (mmol N m^-3)

    if 'R6p' in species:  rates[:,species['R6p']] = dr6p_dt     # Particulate organic phosphate (mmol P m^-3)

    if 'R6s' in species:  rates[:,species['R6s']] = dr6s_dt     # Particulate organic silicate (mmol Si m^-3)

    if 'O3c' in species:  rates[:,species['O3c']] = do3c_dt     # Dissolved inorganic carbon(mg C m^-3)

    if 'O3h' in species:  rates[:,species['O3h']] = do3h_dt     # Total alkalinity (mmol Eq m^-3)

    rates = rates/sec_per_day

    return rates

def state_vars(d3state,num_boxes,species):

    if 'O2o' in species:  o2o = d3state[:,species['O2o']]              # Dissolved oxygen (mg O_2 m^-3)
    else:   o2o = np.zeros(num_boxes)
    
    if 'N1p' in species:  n1p = d3state[:,species['N1p']]              # Phosphate (mmol P m^-3)
    else:   n1p = np.zeros(num_boxes)
    
    if 'N3n' in species:  n3n = d3state[:,species['N3n']]              # Nitrate (mmol N m^-3)
    else:   n3n = np.zeros(num_boxes)

    if 'N4n' in species:  n4n = d3state[:,species['N4n']]              # Ammonium (mmol N m^-3)
    else:   n4n = np.zeros(num_boxes)
    
    if 'O4n' in species:  o4n = d3state[:,species['O4n']]              # Nitrogen sink (mmol N m^-3)
    else:   o4n = np.zeros(num_boxes)

    if 'N5s' in species:  n5s = d3state[:,species['N5s']]              # Silicate (mmol Si m^-3)
    else:   n5s = np.zeros(num_boxes)

    if 'N6r' in species:  n6r = d3state[:,species['N6r']]              # Reduction equivalents (mmol S m^-3)
    else:   n6r = np.zeros(num_boxes)

    if 'B1c' in species:  b1c = d3state[:,species['B1c']]              # Pelagic bacteria carbon (mg C m^-3)
    else:   b1c = np.zeros(num_boxes)

    if 'B1n' in species:  b1n = d3state[:,species['B1n']]              # Pelagic bacteria nitrogen (mmol N m^-3)
    else:   b1n = np.zeros(num_boxes)

    if 'B1p' in species:  b1p = d3state[:,species['B1p']]              # Pelagic bacteria phosphate (mmol P m^-3)
    else:   b1p = np.zeros(num_boxes)

    if 'P1c' in species:  p1c = d3state[:,species['P1c']]             # Diatoms carbon (mg C m^-3)
    else:   p1c = np.zeros(num_boxes)

    if 'P1n' in species:  p1n = d3state[:,species['P1n']]             # Diatoms nitrogen (mmol N m^-3)
    else:   p1n = np.zeros(num_boxes)

    if 'P1p' in species:  p1p = d3state[:,species['P1p']]             # Diatoms phosphate (mmol P m^-3)
    else:   p1p = np.zeros(num_boxes)

    if 'P1l' in species:  p1l = d3state[:,species['P1l']]             # Diatoms chlorophyll (mg Chl-a m^-3)
    else:   p1l = np.zeros(num_boxes)

    if 'P1s' in species:  p1s = d3state[:,species['P1s']]             # Diatoms silicate (mmol Si m^-3) 
    else:   p1s = np.zeros(num_boxes)

    if 'P2c' in species:  p2c = d3state[:,species['P2c']]             # NanoFlagellates carbon (mg C m^-3)
    else:   p2c = np.zeros(num_boxes)

    if 'P2n' in species:  p2n = d3state[:,species['P2n']]             # NanoFlagellates nitrogen (mmol N m^-3)
    else:   p2n = np.zeros(num_boxes)

    if 'P2p' in species:  p2p = d3state[:,species['P2p']]             # NanoFlagellates phosphate (mmol P m^-3)
    else:   p2p = np.zeros(num_boxes)

    if 'P2l' in species:  p2l = d3state[:,species['P2l']]             # NanoFlagellates chlorophyll (mg Chl-a m^-3)
    else:   p2l = np.zeros(num_boxes)

    if 'P3c' in species:  p3c = d3state[:,species['P3c']]             # Picophytoplankton carbon (mg C m^-3)
    else:   p3c = np.zeros(num_boxes)

    if 'P3n' in species:  p3n = d3state[:,species['P3n']]             # Picophytoplankton nitrogen (mmol N m^-3)
    else:   p3n = np.zeros(num_boxes)

    if 'P3p' in species:  p3p = d3state[:,species['P3p']]             # Picophytoplankton phosphate (mmol P m^-3)
    else:   p3p = np.zeros(num_boxes)

    if 'P3l' in species:  p3l = d3state[:,species['P3l']]             # Picophytoplankton chlorophyll (mg Chl-a m^-3)
    else:   p3l = np.zeros(num_boxes)

    if 'P4c' in species:  p4c = d3state[:,species['P4c']]             # Large phytoplankton carbon (mg C m^-3)
    else:   p4c = np.zeros(num_boxes)

    if 'P4n' in species:  p4n = d3state[:,species['P4n']]             # Large phytoplankton nitrogen (mmol N m^-3)
    else:   p4n = np.zeros(num_boxes)

    if 'P4p' in species:  p4p = d3state[:,species['P4p']]             # Large phytoplankton phosphate (mmol P m^-3) 
    else:   p4p = np.zeros(num_boxes)

    if 'P4l' in species:  p4l = d3state[:,species['P4l']]             # Large phytoplankton chlorophyll (mg Chl-a m^-3)
    else:   p4l = np.zeros(num_boxes)

    if 'Z3c' in species:  z3c = d3state[:,species['Z3c']]             # Carnivorous mesozooplankton carbon (mg C m^-3)
    else:   z3c = np.zeros(num_boxes)

    if 'Z3n' in species:  z3n = d3state[:,species['Z3n']]             # Carnivorous mesozooplankton nitrogen (mmol N m^-3)
    else:   z3n = np.zeros(num_boxes)

    if 'Z3p' in species:  z3p = d3state[:,species['Z3p']]             # Carnivorous mesozooplankton phosphate (mmol P m^-3)
    else:   z3p = np.zeros(num_boxes)

    if 'Z4c' in species:  z4c = d3state[:,species['Z4c']]             # Omnivorous mesozooplankton carbon (mg C m^-3)
    else:   z4c = np.zeros(num_boxes)

    if 'Z4n' in species:  z4n = d3state[:,species['Z4n']]             # Omnivorous mesozooplankton nitrogen (mmol N m^-3)
    else:   z4n = np.zeros(num_boxes)

    if 'Z4p' in species:  z4p = d3state[:,species['Z4p']]             # Omnivorous mesozooplankton phosphate (mmol P m^-3)
    else:   z4p = np.zeros(num_boxes)

    if 'Z5c' in species:  z5c = d3state[:,species['Z5c']]             # Microzooplankton carbon (mg C m^-3)
    else:   z5c = np.zeros(num_boxes)

    if 'Z5n' in species:  z5n = d3state[:,species['Z5n']]             # Microzooplankton nitrogen (mmol N m^-3)
    else:   z5n = np.zeros(num_boxes)

    if 'Z5p' in species:  z5p = d3state[:,species['Z5p']]             # Microzooplankton phosphate (mmol P m^-3)
    else:   z5p = np.zeros(num_boxes)

    if 'Z6c' in species:  z6c = d3state[:,species['Z6c']]             # Heterotrophic flagellates carbon (mg C m^-3)
    else:   z6c = np.zeros(num_boxes)

    if 'Z6n' in species:  z6n = d3state[:,species['Z6n']]             # Heterotrophic flagellates nitrogen (mmol N m^-3)
    else:   z6n = np.zeros(num_boxes)

    if 'Z6p' in species:  z6p = d3state[:,species['Z6p']]             # Heterotrophic flagellates phosphate (mmol P m^-3)
    else:   z6p = np.zeros(num_boxes)

    if 'R1c' in species:  r1c = d3state[:,species['R1c']]             # Labile dissolved organic carbon (mg C m^-3)
    else:   r1c = np.zeros(num_boxes)

    if 'R1n' in species:  r1n = d3state[:,species['R1n']]             # Labile dissolved organic nitrogen (mmol N m^-3)
    else:   r1n = np.zeros(num_boxes)

    if 'R1p' in species:  r1p = d3state[:,species['R1p']]             # Labile dissolved organic phosphate (mmol P m^-3)
    else:   r1p = np.zeros(num_boxes)

    if 'R2c' in species:  r2c = d3state[:,species['R2c']]             # Semi-labile dissolved organic carbon (mg C m^-3)
    else:   r2c = np.zeros(num_boxes)

    if 'R3c' in species:  r3c = d3state[:,species['R3c']]             # Semi-refractory Dissolved Organic Carbon (mg C m^-3)
    else:   r3c = np.zeros(num_boxes)

    if 'R6c' in species:  r6c = d3state[:,species['R6c']]             # Particulate organic carbon (mg C m^-3)
    else:   r6c = np.zeros(num_boxes)

    if 'R6n' in species:  r6n = d3state[:,species['R6n']]             # Particulate organic nitrogen (mmol N m^-3)
    else:   r6n = np.zeros(num_boxes)

    if 'R6p' in species:  r6p = d3state[:,species['R6p']]             # Particulate organic phosphate (mmol P m^-3)
    else:   r6p = np.zeros(num_boxes)

    if 'R6s' in species:  r6s = d3state[:,species['R6s']]             # Particulate organic silicate (mmol Si m^-3)
    else:   r6s = np.zeros(num_boxes)

    if 'O3c' in species:  o3c = d3state[:,species['O3c']]             # Dissolved inorganic carbon(mg C m^-3)
    else:   o3c = np.zeros(num_boxes)

    if 'O3h' in species:  o3h = d3state[:,species['O3h']]             # Total alkalinity (mmol Eq m^-3)
    else:   o3h = np.zeros(num_boxes)

    return o2o, n1p, n3n, n4n, o4n, n5s, n6r, b1c, b1n, b1p, \
           p1c, p1n, p1p, p1l, p1s, p2c, p2n, p2p, p2l, p3c, p3n, p3p, p3l, p4c, p4n, p4p, p4l, \
           z3c, z3n, z3p, z4c, z4n, z4p, z5c, z5n, z5p, z6c, z6n, z6p, \
           r1c, r1n, r1p, r2c, r3c, r6c, r6n, r6p, r6s, o3c, o3h   
# pyPOM1D-reducedBFM
One-dimensional Princeton Ocean Model (POM1D) with coupling capabilities with Biogeochemical Flux Model (BFM).
This branch includes the option to perform a skeletal model reduction on BFM using a modified version of the direct relation graph with error propagation method. Species that are identified as unimportant to the model for examining a particular quantity are neglected during the simulation (all unnecessary species and their associated rates are set to zero).

To run the available reduction schemes, set the "REDUCE_BFM" flag to "True" in main_pombfm1d.py. Then, identify the chosen reduction scheme by in the "modified_DRGEP" function located in /reduction/modified_DRGEP.py. The available options, including target and safe tracers, are:

--------------------------------------------------------------------------------------------------------------
Error Function             |   Target Species                          |   Safe Species
--------------------------------------------------------------------------------------------------------------
calc_error_lo_1            |   ['P1l', 'P2l', 'P3l', 'P4l']            |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
calc_error_lo_2            |   ['P2l', 'P3l', 'P4l']                   |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
calc_error_lo_3            |   ['P3l', 'P4l']                          |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
calc_error_lo_4            |   ['P1l', 'P2l', 'P3l', 'P4l']            |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
calc_error_lo_5            |   ['P2l', 'P3l', 'P4l']                   |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
calc_error_lo_6            |   ['P3l', 'P4l']                          |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
calc_error_lo_7            |   ['P1c', 'P2c', 'P3c', 'P4c']            |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
calc_error_lo_8            |   ['P1c', 'P2c', 'P3c', 'P4c']            |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
calc_error_lo_9            |   ['P1c', 'P2c', 'P3c', 'P4c']            |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
--------------------------------------------------------------------------------------------------------------
calc_error_dic_1           |   ['O3c']                                 |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
calc_error_dic_2           |   ['O3c']                                 |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
calc_error_dic_3           |   ['O3c']                                 |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
calc_error_dic_4           |   ['O3c']                                 |   ['N3n', 'N4n', 'P1l', 'P2l', 'P3l', 'P4l']
--------------------------------------------------------------------------------------------------------------
calc_error_pon_1           |   ['P1n', 'P2n', 'P3n', 'P4n', 'R6n']     |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
calc_error_pon_2           |   ['P1n', 'P2n', 'P3n', 'P4n', 'R6n']     |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
calc_error_pon_3           |   ['P1n', 'P2n', 'P3n', 'P4n', 'R6n']     |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
--------------------------------------------------------------------------------------------------------------
calc_error_oxy_1           |   ['O2o']                                 |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
calc_error_oxy_2           |   ['O2o']                                 |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
calc_error_oxy_3           |   ['O2o']                                 |   []
calc_error_oxy_4           |   ['O2o']                                 |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
--------------------------------------------------------------------------------------------------------------
calc_error_in_1            |   ['N1p']                                 |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
calc_error_in_2            |   ['N1p']                                 |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
calc_error_in_3            |   ['N3n']                                 |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
calc_error_in_4            |   ['N3n']                                 |   ['N4n', 'P1l', 'P2l', 'P3l', 'P4l']
--------------------------------------------------------------------------------------------------------------

To use a preset reduction, set the "REDUCE_BFM" flag in main_pombfm1d.py to "False" and identify the preset of choice. The available presets and their respective retained tracers are:

bfm1: {'O2o': 0}
bfm23: {'O2o': 0, 'N1p': 1, 'N3n': 2, 'N4n': 3, 'B1c': 7, 'B1n': 8, 'B1p': 9, 'P1l': 13, 'P2c': 15, 'P2n': 16, 'P2p': 17, 'P2l': 18, 'P3l': 22, 'P4l': 26, 'Z5c': 33, 'Z5p': 35, 'R1c': 39, 'R1n': 40, 'R1p': 41, 'R6c': 44, 'R6n': 45, 'R6p': 46, 'O3c': 48}
bfm34: {'O2o': 0, 'N1p': 1, 'N3n': 2, 'N4n': 3, 'B1c': 7, 'B1n': 8, 'B1p': 9, 'P1c': 10, 'P1n': 11, 'P1p': 12, 'P1l': 13, 'P2c': 15, 'P2n': 16, 'P2p': 17, 'P2l': 18, 'P3c': 19, 'P3n': 20, 'P3p': 21, 'P3l': 22, 'P4c': 23, 'P4n': 24, 'P4p': 25, 'P4l': 26, 'Z5c': 33, 'Z5n': 34, 'Z5p': 35, 'Z6c': 36, 'Z6p': 38, 'R1c': 39, 'R1n': 40, 'R1p': 41, 'R6c': 44, 'R6n': 45, 'R6p': 46}
bfm35:{'O2o': 0, 'N1p': 1, 'N3n': 2, 'N4n': 3, 'B1c': 7, 'B1n': 8, 'B1p': 9, 'P1c': 10, 'P1n': 11, 'P1p': 12, 'P1l': 13, 'P2c': 15, 'P2n': 16, 'P2p': 17, 'P2l': 18, 'P3c': 19, 'P3n': 20, 'P3p': 21, 'P3l': 22, 'P4c': 23, 'P4n': 24, 'P4p': 25, 'P4l': 26, 'Z5c': 33, 'Z5n': 34, 'Z5p': 35, 'Z6c': 36, 'Z6n': 37, 'Z6p': 38, 'R1c': 39, 'R1n': 40, 'R1p': 41, 'R6c': 44, 'R6n': 45, 'R6p': 46}
bfm36: {'O2o': 0, 'N1p': 1, 'N3n': 2, 'N4n': 3, 'B1c': 7, 'B1n': 8, 'B1p': 9, 'P1c': 10, 'P1n': 11, 'P1p': 12, 'P1l': 13, 'P2c': 15, 'P2n': 16, 'P2p': 17, 'P2l': 18, 'P3c': 19, 'P3n': 20, 'P3p': 21, 'P3l': 22, 'P4c': 23, 'P4n': 24, 'P4p': 25, 'P4l': 26, 'Z5c': 33, 'Z5n': 34, 'Z5p': 35, 'Z6c': 36, 'Z6n': 37, 'Z6p': 38, 'R1c': 39, 'R1n': 40, 'R1p': 41, 'R6c': 44, 'R6n': 45, 'R6p': 46, 'O3c': 48}
bfm50: full model


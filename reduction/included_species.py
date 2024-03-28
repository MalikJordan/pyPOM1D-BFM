import numpy as np

def included_species(remaining_species):

    include = {
        'oxygen': True,
        'carbon': True,
        'pel_chem': True,
        'b1': True,
        'p1': True,
        'p2': True,
        'p3': True,
        'p4': True,
        'z3': True,
        'z4': True,
        'z5': True,
        'z6': True,
        'r1': True,
        'r6': True
        }

    # Oxygen
    if 'O2o' not in remaining_species:
        include['oxygen'] = False

    # DIC
    if 'O3c' not in remaining_species:
        include['carbon'] = False

    # Bacteria
    if 'B1c' not in remaining_species and 'B1n' not in remaining_species and 'B1p' not in remaining_species:
        include['b1'] = False

    # Phytoplankton
    if 'P1c' not in remaining_species and 'P1n' not in remaining_species and 'P1p' not in remaining_species and 'P1l' not in remaining_species and 'P1s' not in remaining_species:
        include['p1'] = False
    if 'P2c' not in remaining_species and 'P2n' not in remaining_species and 'P2p' not in remaining_species and 'P2l' not in remaining_species:
        include['p2'] = False
    if 'P3c' not in remaining_species and 'P3n' not in remaining_species and 'P3p' not in remaining_species and 'P3l' not in remaining_species:
        include['p3'] = False
    if 'P4c' not in remaining_species and 'P4n' not in remaining_species and 'P4p' not in remaining_species and 'P4l' not in remaining_species:
        include['p4'] = False

    # Mesozooplankton
    if 'Z3c' not in remaining_species and 'Z3n' not in remaining_species and 'Z3p' not in remaining_species:
        include['z3'] = False
    if 'Z4c' not in remaining_species and 'Z4n' not in remaining_species and 'Z4p' not in remaining_species:
        include['z4'] = False
    
    # Microzooplankton
    if 'Z5c' not in remaining_species and 'Z5n' not in remaining_species and 'Z5p' not in remaining_species:
        include['z5'] = False
    if 'Z6c' not in remaining_species and 'Z6n' not in remaining_species and 'Z6p' not in remaining_species:
        include['z6'] = False

    # Dissolved Organic Matter
    if 'R1c' not in remaining_species and 'R1n' not in remaining_species and 'R1p' not in remaining_species:
        include['r1'] = False

    # Particulate Organic Matter
    if 'R6c' not in remaining_species and 'R6n' not in remaining_species and 'R6p' not in remaining_species and 'R6s' not in remaining_species:
        include['r6'] = False

    # Pelageic Chemistry
    if 'N3n' not in remaining_species and 'N4n' not in remaining_species and 'N6r' not in remaining_species and include['r1'] == False and include['r6'] == False:
        include['pel_chem'] == False

    return include


def remove_species(d3state,d3stateb,species_names,species_removed):

    species = {}
    species_removed_indices = []
    renumber_species = 0
    for index,spec in enumerate(species_names):
        if spec in species_removed:
            species_removed_indices.append(index)
        if spec not in species_removed:
            species[spec] = renumber_species
            renumber_species += 1
    for index in reversed(np.asarray(species_removed_indices)):
        d3state = np.delete(d3state, (index), axis=1)
        d3stateb = np.delete(d3stateb, (index), axis=1)

    return d3state, d3stateb, species

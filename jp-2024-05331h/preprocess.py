import os
import sys

from ase.io import read
from ase.neighborlist import neighbor_list

sys.path.append('..')
from ImpLearn.graft import Graft
from ImpLearn.silanols import Silanols


# Carve Clusters #

slab_file_paths = ['./slab_models/coords_1Kps_{:03d}.xyz'.format(i+1) for i in range(100)]
for file_path in slab_file_paths:
    label = file_path.rsplit('/')[-1].rsplit('.', 1)[0]
    silanols = Silanols(
            file_path,
            #viable_cutoff=6.0,
            viable_cutoff=5.5,
            #file_type='xyz',
            #pbc=[21.01554, 21.01554, 90.23032],
            #bond_cutoffs={('Si', 'O'): 2.3, ('O', 'H'): 1.2},
            #viable_cutoff=5.5,
            #OH_bond_length=0.956,
            #exclude_waters=True,
            #exclude_geminals=True,
            #reorder_podals=True,
            #F_capping=True,
            #reorient_clusters=True,
            #reorder_atoms=True
            )
    print(file_path)
    print('OH Groups')
    print(len(silanols.OH_groups))
    print(silanols.OH_groups)
    print('Geminal OH Groups')
    print(len(silanols.geminal_OH_pairs))
    print(silanols.geminal_OH_pairs)
    print('Vicinal OH Groups')
    print(len(silanols.vicinal_OH_pairs))
    print(silanols.vicinal_OH_pairs)
    print('Viable OH Groups')
    print(len(silanols.viable_OH_pairs))
    print(silanols.viable_OH_pairs)
    silanols.analyze_bonds()
    silanols.analyze_distances('./silanols/' + label + '_d{:s}{:s}.png')
    silanols.save_clusters('./silanols/' + label + '_{:02d}{:02d}.xyz')


# Classify Silanols #

bond_cutoffs = {
        ('Si', 'Si'): 2.0, ('O', 'O'): 2.0, ('Si', 'O'): 2.3, ('O', 'H'): 1.2,
        ('F', 'F'): 2.0, ('O', 'F'): 2.0, ('Si', 'F'): 2.3, ('F', 'H'): 1.2,
        }

vicinal_atoms = ['F', 'F', 'F', 'F', 'Si', 'Si', 'O', 'O', 'O', 'H', 'H']
nonvicinal_atoms = ['F', 'F', 'F', 'F', 'F', 'F', 'Si', 'Si', 'O', 'O', 'H', 'H']

silanol_file_paths = ['./silanols/' + file_name for file_name in os.listdir('./silanols') if file_name.endswith('.xyz')]
for silanol_file_path in silanol_file_paths:
    cluster = read(silanol_file_path, 0, 'xyz')
    atoms = cluster.get_chemical_symbols()
    bonds = neighbor_list('ij', cluster, bond_cutoffs)
    dangling = False
    for i, X in enumerate(atoms):
        i_neighbors = bonds[1][bonds[0] == i]
        if X == 'Si' and len(i_neighbors) != 4:
            dangling = True
            break
        elif X == 'O' and len(i_neighbors) != 2:
            dangling = True
            break
        elif X == 'F' and len(i_neighbors) != 1:
            dangling = True
            break
        elif X == 'H' and len(i_neighbors) != 1:
            dangling = True
            break
    if atoms == vicinal_atoms and not dangling:
        os.system('mv ' + silanol_file_path + ' ./silanols_vicinal/')
    elif atoms == nonvicinal_atoms and not dangling:
        os.system('mv ' + silanol_file_path + ' ./silanols_nonvicinal/')
    else:
        os.system('mv ' + silanol_file_path + ' ./silanols_other/')


# Graft Chromium Complexes #

silanol_file_paths = ['./silanols_vicinal/' + file_name for file_name in os.listdir('./silanols_vicinal') if file_name.endswith('.xyz')]
silanol_labels = [file_path.rsplit('/')[-1].rsplit('.')[0] for file_path in silanol_file_paths]
silanol_clusters = [read(file_path, 0, 'xyz') for file_path in silanol_file_paths]

ref_file_paths = [
        './templates/1I.xyz',
        './templates/1IIa.xyz',
        './templates/1IIb.xyz',
        './templates/1III.xyz',
        './templates/1IV.xyz',
        './templates/1TSa-III-IV.xyz',
        './templates/1TSb-III-IV.xyz',
        './templates/4XIa.xyz',
        './templates/4XIIa.xyz',
        './templates/4XIIIa.xyz',
        './templates/4TSa-XII-XIII.xyz',
        ]
ref_labels = [
        '1I',
        '1IIa',
        '1IIb',
        '1III',
        '1IV',
        '1TSa-III-IV',
        '1TSb-III-IV',
        '4XIa',
        '4XIIa',
        '4XIIIa',
        '4TSa-XII-XIII',
        ]
ref_clusters = [read(file_path, 0, 'xyz') for file_path in ref_file_paths]

graft_file_paths = [['./graft_vicinal/' + slabel + '.' + rlabel + '.xyz' for rlabel in ref_labels] for slabel in silanol_labels]
for i, _ in enumerate(silanol_file_paths):
    for j, _ in enumerate(ref_file_paths):
        if not os.path.exists(graft_file_paths[i][j]):
            graft = Graft(
                    silanol_clusters[i],
                    ref_clusters[j],
                    podal_atoms=[0, 1, 2, 3, 4, 5, 6],
                    match_atoms=[4, 5, 7, 8]
                    )
            graft.run()
            graft.save_cluster(graft_file_paths[i][j])


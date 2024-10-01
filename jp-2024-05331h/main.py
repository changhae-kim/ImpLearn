import joblib
import numpy
import os
import sys

from ase import Atoms
from ase.io import read, write

sys.path.append('..')
from ImpLearn.crest import Crest
from ImpLearn.gaussian import Gaussian
from ImpLearn.graft import Graft
from ImpLearn.regression import Kernel
from ImpLearn.sampler import Sampler
from ImpLearn.xtb import xTB

from ImpLearn.gaussian_tools import write_clusters, read_geom_opt, read_thermochem
from ImpLearn.silanols_tools import permute_podal_atoms

from tools import calc_Kg, calc_k2_K1, calc_kg, calc_ac_rate, calc_dc_rate, calc_pdc_rate, get_sxc


initial_batch_size = 20
initial_batch = [
        # Initial Batch 20 Sites #
        # 54, 124, 263, 175, 282, 213, 303, 281, 100, 196, 142, 37, 122, 377, 246, 59, 68, 168, 254, 52,
        # Combined Model 248 Sites #
        # 0, 3, 4, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 30, 31, 32, 34, 35, 36, 37, 39, 40, 42, 44, 45, 46, 51, 52, 54, 56, 58, 59, 60, 61, 62, 64, 68, 70, 71, 73, 74, 79, 80, 81, 83, 87, 88, 90, 91, 95, 98, 100, 101, 103, 104, 105, 106, 109, 112, 115, 117, 121, 122, 123, 124, 125, 126, 127, 128, 131, 132, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 148, 150, 151, 154, 155, 156, 157, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 174, 175, 178, 179, 180, 181, 184, 186, 187, 188, 191, 192, 193, 195, 196, 197, 198, 199, 202, 203, 204, 205, 206, 208, 209, 210, 212, 213, 216, 217, 221, 222, 223, 225, 227, 228, 229, 230, 232, 233, 234, 235, 241, 242, 243, 245, 246, 249, 250, 253, 254, 256, 257, 263, 264, 265, 266, 267, 270, 273, 274, 275, 276, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 292, 294, 295, 297, 298, 299, 301, 303, 305, 306, 307, 313, 314, 315, 316, 317, 318, 319, 321, 322, 323, 325, 326, 328, 329, 331, 333, 335, 336, 337, 341, 342, 343, 344, 345, 346, 348, 352, 353, 354, 355, 356, 358, 360, 361, 363, 365, 366, 367, 368, 369, 372, 373, 374, 375, 377, 379, 380, 381, 382, 383, 384, 386, 387,
        # Most Active 10 Sites #
        # 27, 46, 121, 128, 181, 233, 306, 353, 375, 386,
        # Random 10 Sites #
        # 54, 100, 124, 175, 196, 213, 263, 281, 282, 303,
        ]
pick_struc = [
        # Grafting Steps
        # 0, 1, 2, 3, 4
        # Catalytic Steps
        # 7, 8, 9, 10
        # Importance Learning
        # 3, 5, 6, 7, 8, 10
        3, 5, 6, 7, 8, 10
        ]
exclude_clusters = [
        # No Candidate Conformer 3 Sites #
        # 111, 309, 312,
        ]


if len(sys.argv) > 3:
    prep_temp = float(sys.argv[1])
    max_iter = int(sys.argv[2])
    pass_point = sys.argv[3]
else:
    print('Usage')
    print('python {:s} [Prep Temp] [# Iter] [Pass Points]'.format(sys.argv[0]))
    print('')
    print('Pass Points')
    print('c -- crest')
    print('x -- xtb')
    print('g -- gaussian')
    print('k -- kernel')
    exit()

assert prep_temp in [ 473.15, 573.15, 673.15, 773.15, 873.15, 973.15, 1073.15 ]


# Import Silanols & Chromium Complexes #

silanol_file_paths = ['./silanols_vicinal/' + file_name for file_name in sorted(os.listdir('./silanols_vicinal')) if file_name.endswith('.xyz')]
silanol_labels = [file_path.rsplit('/', 1)[-1].rsplit('.', 1)[0] for file_path in silanol_file_paths]
silanol_clusters = [read(file_path, 0, 'xyz') for file_path in silanol_file_paths]

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
ref_labels = [ref_labels[s] for s in pick_struc]

index = {n: m for m, n in enumerate(pick_struc)}

graft_file_paths = [['./graft_vicinal/' + slabel + '.' + rlabel + '.xyz' for rlabel in ref_labels] for slabel in silanol_labels]


# Set Up Thermodynamics #

prep_time = 3600.0
temp = 373.15
pressure = 1.0
vib_cutoff = 100.0

E_c, H_c, S_c, G_c = read_thermochem('./templates/wb97xd_tzvp/CrO2Cl2.log', temp=prep_temp, pressure=pressure, trans=True, rot=True, vib_cutoff=vib_cutoff, gibbs_only=False)
E_d, H_d, S_d, G_d = read_thermochem('./templates/wb97xd_tzvp/HCl.log', temp=prep_temp, pressure=pressure, trans=True, rot=True, vib_cutoff=vib_cutoff, gibbs_only=False)
E_a, H_a, S_a, G_a = read_thermochem('./templates/wb97xd_tzvp/C2H4.log', temp=temp, pressure=pressure, trans=True, rot=True, vib_cutoff=vib_cutoff, gibbs_only=False)


# Calculate Coordinates #

podal_atoms = [0, 1, 2, 3]
podal_coords = [cluster.get_positions()[podal_atoms] for cluster in silanol_clusters]
sxcs = [get_sxc(coords) for coords in podal_coords]


# Initialize Sampler #

sampler = Sampler(
        len(silanol_clusters),
        initial_batch_size=initial_batch_size,
        ##initial_batch=initial_batch,
        batch_size=1,
        exclude=exclude_clusters,
        random_state=0
        )

features = []
targets = []


# Iterate Importance Learning #

for i in range(max_iter+1):

    if i == 0:
        samples = [j for j in sampler.samples]
    else:
        samples = sampler.sample(weights)

    print('Batch {:d}'.format(i))
    for j in samples:
        print(silanol_labels[j], j)

    for j in samples:


        # Run CREST #

        spins = [0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3]
        crest_constraints = [
                '$constrain\n  atoms: 1-4\n  force constant=0.5\n  reference=coord.ref\n$metadyn\n  atoms: 5-11\n$end\n',
                '$constrain\n  atoms: 1-4\n  force constant=0.5\n  reference=coord.ref\n$metadyn\n  atoms: 5-14\n$end\n',
                '$constrain\n  atoms: 1-4\n  force constant=0.5\n  reference=coord.ref\n$metadyn\n  atoms: 5-14\n$end\n',
                '$constrain\n  atoms: 1-4\n  force constant=0.5\n  reference=coord.ref\n$metadyn\n  atoms: 5-17\n$end\n',
                '$constrain\n  atoms: 1-4\n  force constant=0.5\n  reference=coord.ref\n$metadyn\n  atoms: 5-12\n$end\n',
                '$constrain\n  atoms: 1-4\n  force constant=0.5\n  reference=coord.ref\n  distance: 10, 13, 2.421\n  distance: 14, 13, 2.358\n  distance: 10, 9, 2.017\n  distance: 14, 9, 2.067\n$metadyn\n  atoms: 5-7, 8, 11-12, 15-16, 17\n$end\n',
                '$constrain\n  atoms: 1-4\n  force constant=0.5\n  reference=coord.ref\n  distance: 10, 17, 2.374\n  distance: 14, 17, 2.380\n  distance: 10, 8, 2.032\n  distance: 14, 8, 2.063\n$metadyn\n  atoms: 5-7, 9, 11-12, 13, 15-16\n$end\n',
                '$constrain\n  atoms: 1-4\n  force constant=0.5\n  reference=coord.ref\n$metadyn\n  atoms: 5-17\n$end\n',
                '$constrain\n  atoms: 1-4\n  force constant=0.5\n  reference=coord.ref\n  distance: 10, 11, 2.444\n$metadyn\n  atoms: 5-9, 12-23\n$end\n',
                '$constrain\n  atoms: 1-4\n  force constant=0.5\n  reference=coord.ref\n$metadyn\n  atoms: 5-23\n$end\n',
                '$constrain\n  atoms: 1-4\n  force constant=0.5\n  reference=coord.ref\n  distance: 10, 11, 2.041\n  distance: 11, 12, 1.413\n  distance: 12, 13, 2.121\n  distance: 10, 13, 2.138\n$metadyn\n  atoms: 5-9, 14-23\n$end\n',
                ]
        xtb_constraints = [
                '$fix\n  atoms: 1-4\n$end\n',
                '$fix\n  atoms: 1-4\n$end\n',
                '$fix\n  atoms: 1-4\n$end\n',
                '$fix\n  atoms: 1-4\n$end\n',
                '$fix\n  atoms: 1-4\n$end\n',
                '$fix\n  atoms: 1-4\n$constrain\n  force constant=0.5\n  distance: 10, 13, 2.421\n  distance: 14, 13, 2.358\n  distance: 10, 9, 2.017\n  distance: 14, 9, 2.067\n$end\n',
                '$fix\n  atoms: 1-4\n$constrain\n  force constant=0.5\n  distance: 10, 17, 2.374\n  distance: 14, 17, 2.380\n  distance: 10, 8, 2.032\n  distance: 14, 8, 2.063\n$end\n',
                '$fix\n  atoms: 1-4\n$end\n',
                '$fix\n  atoms: 1-4\n$end\n',
                '$fix\n  atoms: 1-4\n$end\n',
                '$fix\n  atoms: 1-4\n$constrain\n  force constant=0.5\n  distance: 10, 11, 2.041\n  distance: 11, 12, 1.413\n  distance: 12, 13, 2.121\n  distance: 10, 13, 2.138\n$end\n',
                ]
        exclude_atoms = [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3, 8, 9, 12, 13],
                [0, 1, 2, 3, 7, 9, 13, 16],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3, 9, 10, 11, 12],
                ]
        exclude_elements = [[], [], [], [], [], [], [], ['H'], ['H'], ['H'], ['H']]

        spins = [spins[s] for s in pick_struc]
        crest_constraints = [crest_constraints[s] for s in pick_struc]
        xtb_constraints = [xtb_constraints[s] for s in pick_struc]
        exclude_atoms = [exclude_atoms[s] for s in pick_struc]
        exclude_elements = [exclude_elements[s] for s in pick_struc]

        crest = Crest(
                graft_file_paths[j], './crest_vicinal',
                charges=0, spins=spins,
                n_proc=24,
                constraints=crest_constraints,
                #charges=0, spins=3,
                #n_proc=24,
                #constraints='$constrain\n  atoms: 1-4\n  force constant=0.5\n  reference=coord.ref\n$metadyn\n  atoms: 5-11\n$end\n'
                )
        crest.setup()
        crest.run(dry_run=True)

        if i == max_iter and 'c' not in pass_point:
            continue

        regraft_file_paths = ['./crest_vicinal/' + label + '.xyz' for label in crest.labels]
        regraft_clusters = [[] for _ in crest.conformers]
        for n, _ in enumerate(crest.conformers):
            if not os.path.exists(regraft_file_paths[n]):
                for m, _ in enumerate(crest.conformers[n]):
                    atoms = crest.conformers[n][m].get_chemical_symbols()
                    coords = crest.conformers[n][m].get_positions()
                    origin = 0.5 * (coords[4] + coords[5])
                    axes = numpy.empty((3, 3))
                    axes[0] = coords[5] - coords[4]
                    axes[0] = axes[0] / numpy.linalg.norm(axes[0])
                    axes[2] = coords[7] + coords[8] - coords[4] - coords[5]
                    axes[2] = axes[2] - axes[0] * numpy.inner(axes[0], axes[2])
                    axes[2] = axes[2] / numpy.linalg.norm(axes[2])
                    axes[1] = numpy.cross(axes[2], axes[0])
                    coords = numpy.einsum('jk,ik->ij', axes, coords-origin)
                    cluster = Atoms(atoms, coords)
                    graft = Graft(
                            silanol_clusters[j],
                            cluster,
                            podal_atoms=[0, 1, 2, 3]
                            )
                    graft.run()
                    regraft_clusters[n].append(graft.output_cluster)
                write(regraft_file_paths[n], regraft_clusters[n], 'xyz', append=True)
            else:
                regraft_clusters[n] = read(regraft_file_paths[n], ':', 'xyz')


        # Run xTB #

        xtb = xTB(
                regraft_file_paths, './xtb_vicinal',
                charges=0, spins=spins,
                n_proc=24, method='GFN2-xTB',
                constraints=xtb_constraints,
                opt_thresh='vtight',
                e_window=0.00956161, r_thresh=0.125,
                exclude_atoms=exclude_atoms,
                exclude_elements=exclude_elements,
                degeneracies=[[g for g in degeneracies] for degeneracies in crest.degeneracies]
                #charges=0, spins=3,
                #n_proc=24, method='GFN2-xTB',
                #constraints='$fix\n  atoms: 1-4\n$constrain\n  force constant=0.5\n  distance: 10, 11, auto\n$end\n'
                #opt_thresh='vtight',
                #e_window=0.009561608625843094, r_thresh=0.125,
                #exclude_atoms=[0, 1, 2, 3],
                #exclude_elements='H',
                #degeneracies=1
                )
        xtb.setup()
        xtb.run(dry_run=True)

        if i == max_iter and 'x' not in pass_point:
            continue

        xtb.sort_conformers()

        xtb_file_paths = [workspaces[0].rsplit('.', 1)[0] + '.xyz' for workspaces in xtb.workspaces]
        for n, _ in enumerate(xtb.workspaces):
            if not os.path.exists(xtb_file_paths[n]):
                write(xtb_file_paths[n], xtb.clusters[n], 'xyz', append=True)


        # Run Gaussian #

        struct_types = ['EQ', 'EQ', 'EQ', 'EQ', 'EQ', 'TS', 'TS', 'EQ', 'EQ', 'EQ', 'TS']
        mults = [1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4]
        temps = [473.15, 473.15, 473.15, 473.15, 473.15, 473.15, 473.15, 373.15, 373.15, 373.15, 373.15]
        gen_basis = [
                'Si O H 0\nTZVP\n****\nF 0\n3-21G\n****',
                'Cr 0\nDef2TZVP\n****\nSi O Cl H 0\nTZVP\n****\nF 0\n3-21G\n****',
                'Cr 0\nDef2TZVP\n****\nSi O Cl H 0\nTZVP\n****\nF 0\n3-21G\n****',
                'Cr 0\nDef2TZVP\n****\nSi O Cl 0\nTZVP\n****\nF 0\n3-21G\n****',
                'Cr 0\nDef2TZVP\n****\nSi O 0\nTZVP\n****\nF 0\n3-21G\n****',
                'Cr 0\nDef2TZVP\n****\nSi O Cl 0\nTZVP\n****\nF 0\n3-21G\n****',
                'Cr 0\nDef2TZVP\n****\nSi O Cl 0\nTZVP\n****\nF 0\n3-21G\n****',
                'Cr 0\nDef2TZVP\n****\nSi O C H 0\nTZVP\n****\nF 0\n3-21G\n****',
                'Cr 0\nDef2TZVP\n****\nSi O C H 0\nTZVP\n****\nF 0\n3-21G\n****',
                'Cr 0\nDef2TZVP\n****\nSi O C H 0\nTZVP\n****\nF 0\n3-21G\n****',
                'Cr 0\nDef2TZVP\n****\nSi O C H 0\nTZVP\n****\nF 0\n3-21G\n****',
                ]
        ts_criteria = [
                {},
                {},
                {},
                {},
                {},
                {(9, 12): (2.3, 2.6), (13, 12): (2.3, 2.6), (9, 8): (1.9, 2.2), (13, 8): (1.9, 2.2), (5, 8): (1.6, 1.7), (4, 7): (1.6, 1.7)},
                {(9, 16): (2.3, 2.6), (13, 16): (2.3, 2.6), (9, 7): (1.9, 2.2), (13, 7): (1.9, 2.2), (4, 7): (1.6, 1.7), (5, 8): (1.6, 1.7)},
                {},
                {},
                {},
                {(9, 10): (1.9, 2.2), (11, 12): (2.0, 2.3), (9, 12): (1.9, 2.2)},
                ]

        struct_types = [struct_types[s] for s in pick_struc]
        mults = [mults[s] for s in pick_struc]
        temps = [temps[s] for s in pick_struc]
        gen_basis = [gen_basis[s] for s in pick_struc]
        ts_criteria = [ts_criteria[s] for s in pick_struc]

        gaussian_prefixes = ['./gaussian_vicinal_wb97xd_tzvp_f321g/' + label for label in crest.labels]
        cp_prefixes = ['./gaussian_vicinal_wb97xd_tzvp_f321g/bsse_sp/' + label for label in crest.labels]
        gaussian = Gaussian(
                xtb_file_paths, gaussian_prefixes,
                which=':', file_type='xyz',
                struct_types=struct_types,
                charges=0, mults=mults,
                temps=temps, pressures=1.0, vib_cutoff=vib_cutoff,
                n_proc=24, method='wB97XD/Gen',
                gen_basis=gen_basis,
                opt_thresh='Normal',
                frozen_atoms=[0, 1, 2, 3],
                ts_criteria=ts_criteria,
                e_window=0.00956161, r_thresh=0.125,
                exclude_atoms=[0, 1, 2, 3],
                exclude_elements=exclude_elements,
                degeneracies=[[g for g in degeneracies] for degeneracies in xtb.degeneracies],
                cp_prefixes=cp_prefixes
                #which=':', file_type='xyz',
                #struct_types='EQ',
                #charges=0, mults=4,
                #temps=373.15, pressures=1.0, vib_cutoff=100.0,
                #n_proc=24, method='wB97XD/Gen',
                #gen_basis='Cr 0\nDef2TZVP\n****\nSi O C H 0\nTZVP\n****\nF 0\n3-21G\n****',
                #opt_thresh='Normal',
                #frozen_atoms=[0, 1, 2, 3],
                #ts_criteria={(10, 11): (1.9, 2.4), (10, 13): (1.9, 2.4), (12, 13): (1.9, 2.4)},
                #e_window=0.00956161, r_thresh=0.125,
                #exclude_atoms=[0, 1, 2, 3],
                #exclude_elements='H',
                #degeneracies=1,
                #cp_prefixes=None
                )
        gaussian.setup()
        gaussian.run(dry_run=True)

        if i == max_iter and 'g' not in pass_point:
            continue

        gaussian.sort_conformers()

        gaussian_file_paths = [prefix + '.xyz' for prefix in gaussian_prefixes]
        for n, _ in enumerate(gaussian.optimizers):
            if not os.path.exists(gaussian_file_paths[n]):
                comments = [ 'degeneracy: {:d}, energy: {:.8f}'.format(gaussian.degeneracies[n][m], gaussian.energies[n][m]) for m, _ in enumerate(gaussian.clusters[n]) ]
                write_clusters(gaussian_file_paths[n], gaussian.clusters[n], comments)


        # Chemical Kinetics #

        kB = 1.380649e-23 / 4.3597447222071e-18
        T = temp

        g = gaussian.degeneracies
        G = gaussian.get_gibbs_energies()
        dE = gaussian.cp_energies

        #k2, K1 = calc_k2_K1( temp, G_a, g[index[7]], G[index[7]], g[index[8]], G[index[8]], g[index[10]], G[index[10]] )

        k2, _ = calc_k2_K1( temp, G_a, g[index[7]], G[index[7]], g[index[8]], G[index[8]], g[index[10]], G[index[10]] )
        _, K1 = calc_k2_K1( temp, G_a, g[index[7]], G[index[7]], g[index[8]], numpy.add(G[index[8]], dE[index[8]]), g[index[10]], G[index[10]] )

        ac_rate = calc_ac_rate(temp, pressure, k2, K1)
        dc_rate = calc_dc_rate(temp, pressure)
        pdc_rate = calc_pdc_rate(temp, pressure, k2, K1)
        isr_rate = 1.0 / ( 1.0/ac_rate + 1.0/dc_rate )

        if prep_temp == 1e+99:
            kg = 1e+99
            prob = 1.0
        else:
            kg = calc_kg( prep_temp, g[index[3]], G[index[3]], numpy.concatenate((g[index[5]], g[index[6]])), numpy.concatenate((G[index[5]], G[index[6]])) )
            prob = -numpy.expm1(-kg*prep_time)

        eff_rate = pdc_rate * prob
        sim_rate = isr_rate * prob

        print('{:s} {:e} {:e} {:e} {:e} {:e} {:e} {:e}'.format(os.path.basename(silanol_file_paths[j][:-4]), k2, K1, ac_rate, isr_rate, kg, prob, sim_rate))


        # Symmetrize Exterior Coordinates #

        permutes = permute_podal_atoms('vicinal')
        for permute in permutes:
            sxc = get_sxc(podal_coords[j][permute])
            features.append(sxc)
            targets.append([ numpy.log(k2), numpy.log(K1), numpy.log(kg) ])
        reflect = [1, 0, 3, 2]
        coords = numpy.copy(podal_coords[j])
        coords[:, 1] = -coords[:, 1]
        coords = coords[reflect]
        for permute in permutes:
            sxc = get_sxc(coords[permute])
            features.append(sxc)
            targets.append([ numpy.log(k2), numpy.log(K1), numpy.log(kg) ])
        permutes = permutes + [[permute[i] for i in reflect] for permute in permutes]


    # Stopping Point #

    if i == max_iter and 'k' not in pass_point:
        continue


    # Kernel Regression #

    kernel_file_path = './kernel_vicinal_wb97xd_tzvp_f321g_prep{:.0f}K/iter{:03d}_batch{:03d}.dump'.format(prep_temp, i, len(sampler.samples))
    if not os.path.exists(kernel_file_path):
        kernels = []
        X = numpy.array(features)
        Y = numpy.array(targets)
        for n, _ in enumerate(targets[0]):
            kernel = Kernel(random_state=0)
            kernel.fit(X, Y[:, n])
            kernels.append(kernel)
        joblib.dump(kernels, kernel_file_path)
    else:
        kernels = joblib.load(kernel_file_path)


    # Predict on Ensemble #

    X = numpy.array(sxcs)
    Y = []
    for kernel in kernels:
        Y.append(kernel.predict(X))
    #U = calc_pdc_rate(temp, pressure, numpy.exp(Y[0]), numpy.exp(Y[1]))
    U = 1.0 / ( 1.0/calc_ac_rate(temp, pressure, numpy.exp(Y[0]), numpy.exp(Y[1])) + 1.0/calc_dc_rate(temp, pressure) )
    V = -numpy.expm1(-numpy.exp(Y[2])*prep_time)
    Z = U * V
    weights = Z / numpy.sum(Z)

    print('{:e} {:e} {:e}'.format(U.min(), U.mean(), U.max()))
    print('{:e} {:e} {:e}'.format(V.min(), V.mean(), V.max()))
    print('{:e} {:e} {:e}'.format(Z.min(), Z.mean(), Z.max()))


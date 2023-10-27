import numpy
import os

from ase import Atoms
from ase.io import read

from .gaussian_tools import check_normal_termination, check_geometry, read_geom_opt, read_orbitals, read_thermochem
#from .graft import MatchCoords


class Gaussian():

    def __init__(self, file_paths, prefixes,
            which=':', file_type='xyz',
            struc_types='EQ',
            charges=0, mults=4,
            temps=373.15, pressures=1.0, vib_cutoff=100.0,
            n_proc=24, method='wB97XD/Gen',
            gen_basis='Cr 0\nDef2TZVP\n****\nSi O C H 0\nTZVP\n****\nF 0\n3-21G\n****',
            opt_thresh='Normal',
            frozen_atoms=[0, 1, 2, 3],
            ts_criteria={(10, 11): (1.9, 2.4), (10, 13): (1.9, 2.4), (12, 13): (1.9, 2.4)},
            e_window=0.00956161, r_thresh=0.125,
            exclude_atoms=[0, 1, 2, 3],
            exclude_elements='H',
            degeneracies=1,
            add_prefixes=None
            ):

        if isinstance(which, int):
            self.structures = [[read(file_path, which, file_type)] for file_path in file_paths]
        elif isinstance(which, str):
            self.structures = [read(file_path, which, file_type) for file_path in file_paths]
        elif isinstance(which, list):
            self.structures = [read(file_path, ':', file_type) for file_path in file_paths]
            if isinstance(which[0], int):
                self.structures = [[structures[i] for i in which] for structures in self.structures]
            elif isinstance(which[0], list):
                self.structures = [[structures[i] for i in whi] for whi, structures in zip(which, self.structures)]
            else:
                exit('not sure which structures to take')
        else:
            exit('not sure which structures to take')

        self.prefixes = prefixes

        n_struct = len(self.structures)

        if isinstance(struc_types, str):
            self.struc_types = [struc_types] * n_struct
        else:
            self.struc_types = struc_types

        if isinstance(charges, int):
            self.charges = [charges] * n_struct
        else:
            self.charges = charges

        if isinstance(mults, int):
            self.mults = [mults] * n_struct
        else:
            self.mults = mults

        if isinstance(temps, float):
            self.temps = [temps] * n_struct
        else:
            self.temps = temps

        if isinstance(pressures, float):
            self.pressures = [pressures] * n_struct
        else:
            self.pressures = pressures

        self.vib_cutoff = vib_cutoff

        self.n_proc = n_proc
        self.method = method

        if isinstance(gen_basis, str):
            self.gen_basis = [gen_basis] * n_struct
        else:
            self.gen_basis = gen_basis

        self.opt_thresh = opt_thresh

        if frozen_atoms == [] or isinstance(frozen_atoms[0], int):
            self.frozen_atoms = [frozen_atoms] * n_struct
        else:
            self.frozen_atoms = frozen_atoms

        if isinstance(ts_criteria, dict):
            self.ts_criteria = [ts_criteria] * n_struct
        else:
            self.ts_criteria = ts_criteria

        self.e_window = e_window
        self.r_thresh = r_thresh

        if isinstance(exclude_atoms, int):
            self.exclude_atoms = [[exclude_atoms]] * n_struct
        elif exclude_atoms == [] or isinstance(exclude_atoms[0], int):
            self.exclude_atoms = [exclude_atoms] * n_struct
        else:
            self.exclude_atoms = exclude_atoms

        if isinstance(exclude_elements, str):
            self.exclude_elements = [[exclude_elements]] * n_struct
        elif exclude_elements == [] or isinstance(exclude_elements[0], str):
            self.exclude_elements = [exclude_elements] * n_struct
        else:
            self.exclude_elements = exclude_elements

        if isinstance(degeneracies, int):
            self.degeneracies = [[degeneracies] for i in range(n_struct)]
        elif isinstance(degeneracies[0], int):
            self.degeneracies = [[g for g in degeneracies] for i in range(n_struct)]
        else:
            self.degeneracies = degeneracies

        self.optimizers = [[] for i in range(n_struct)]
        self.energies = [[] for i in range(n_struct)]
        self.clusters = [[] for i in range(n_struct)]
        self.enthalpies = [[] for i in range(n_struct)]
        self.entropies = [[] for i in range(n_struct)]
        self.gibbs_energies = [[] for i in range(n_struct)]
        self.orbitals = [[] for i in range(n_struct)]

        self.add_prefixes = add_prefixes

        self.add_optimizers = [[] for i in range(n_struct)]
        self.add_energies = [[] for i in range(n_struct)]

        return

    def setup(self):
        n_struct = len(self.structures)
        for i in range(n_struct):
            n_digits = str(len(str(len(self.structures[i]))))
            for j, _ in enumerate(self.structures[i]):
                optimizer = ('{:s}.{:0' + n_digits + 'd}').format(self.prefixes[i], j)
                if self.struc_types[i].upper() == 'TS':
                    self.setup_ts_opt(i, optimizer, self.structures[i][j])
                if self.struc_types[i].upper() == 'CONST':
                    self.setup_const_opt(i, optimizer, self.structures[i][j])
                else:
                    self.setup_geom_opt(i, optimizer, self.structures[i][j])
                if optimizer not in self.optimizers[i]:
                    self.optimizers[i].append(optimizer)
        if self.add_prefixes is not None:
            self.add_setup()
        return

    def add_setup(self):
        n_struct = len(self.structures)
        for i in range(n_struct):
            n_digits = str(len(str(len(self.structures[i]))))
            for j, _ in enumerate(self.structures[i]):
                optimizer = ('{:s}.{:0' + n_digits + 'd}').format(self.add_prefixes[i], j)
                if optimizer not in self.add_optimizers[i]:
                    self.add_optimizers[i].append(optimizer)
        return

    def setup_geom_opt(self, state, optimizer, cluster):

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()

        if self.opt_thresh == 'NoOpt':
            opt = ''
        elif self.opt_thresh == 'Loose':
            opt = 'Int=SG1 Opt=(Loose,MaxCycles=200) '
        elif self.opt_thresh == 'Tight':
            opt = 'Int=UltraFine Opt=(Tight,MaxCycles=200) '
        elif self.opt_thresh == 'VeryTight':
            opt = 'Int=UltraFine Opt=(VeryTight,MaxCycles=200) '
        else:
            opt = 'Opt=(MaxCycles=200) '

        if self.temps[state] == 0.0 or self.pressures[state] == 0.0:
            freq = ''
        else:
            freq = 'Freq Temp={temp:.3f} Pressure={pressure:.5f}'.format(temp=self.temps[state], pressure=self.pressures[state])

        header = '''%NProcShared={n_proc:d}
#n {method:s} NoSymm SCF=XQC {opt:s}{freq:s}

 {label:s}

{charge:d} {mult:d}
'''.format(n_proc=self.n_proc, method=self.method, opt=opt, freq=freq, label=os.path.basename(optimizer), charge=self.charges[state], mult=self.mults[state])
        body = ''
        for j, (X, coord) in enumerate(zip(atoms, coords)):
            if j in self.frozen_atoms[state]:
                atom_type = -1
            else:
                atom_type = 0
            body += '{X:2s} {t:2d} {x:9f} {y:9f} {z:9f}\n'.format(X=X, t=atom_type, x=coord[0], y=coord[1], z=coord[2])
        footer = '\n'
        if self.gen_basis[state] != '':
            footer += self.gen_basis[state] + '\n\n'

        if not os.path.exists('{:s}.com'.format(optimizer)):
            f = open('{:s}.com'.format(optimizer), 'wt')
            f.write(header + body + footer)
            f.close()

        return

    def setup_const_opt(self, state, optimizer, cluster):

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()

        if self.opt_thresh == 'NoOpt':
            opt = ''
        elif self.opt_thresh == 'Loose':
            opt = 'Int=SG1 Opt=(Loose,GIC,ModRedundant,MaxCycles=200) '
        elif self.opt_thresh == 'Tight':
            opt = 'Int=UltraFine Opt=(Tight,GIC,ModRedundant,MaxCycles=200) '
        elif self.opt_thresh == 'VeryTight':
            opt = 'Int=UltraFine Opt=(VeryTight,GIC,ModRedundant,MaxCycles=200) '
        else:
            opt = 'Opt=(GIC,ModRedundant,MaxCycles=200) '

        if self.temps[state] == 0.0 or self.pressures[state] == 0.0:
            freq = ''
        else:
            freq = 'Freq Temp={temp:.3f} Pressure={pressure:.5f}'.format(temp=self.temps[state], pressure=self.pressures[state])

        header = '''%NProcShared={n_proc:d}
#n {method:s} NoSymm SCF=XQC {opt:s}{freq:s}

 {label:s}

{charge:d} {mult:d}
'''.format(n_proc=self.n_proc, method=self.method, opt=opt, freq=freq, label=os.path.basename(optimizer), charge=self.charges[state], mult=self.mults[state])
        body = ''
        for j, (X, coord) in enumerate(zip(atoms, coords)):
            if j in self.frozen_atoms[state]:
                atom_type = -1
            else:
                atom_type = 0
            body += '{X:2s} {t:2d} {x:9f} {y:9f} {z:9f}\n'.format(X=X, t=atom_type, x=coord[0], y=coord[1], z=coord[2])
        footer = '\n'
        if self.gen_basis[state] != '':
            footer += self.gen_basis[state] + '\n\n'

        if not os.path.exists('{:s}.com'.format(optimizer)):
            f = open('{:s}.com'.format(optimizer), 'wt')
            f.write(header + body + footer)
            f.close()

        return

    def setup_ts_opt(self, state, optimizer, cluster):

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()

        if self.opt_thresh == 'NoOpt':
            opt = ''
        elif self.opt_thresh == 'Loose':
            opt = 'Int=SG1 Opt=(Loose,TS,CalcFC,NoEigen,MaxCycles=200) '
        elif self.opt_thresh == 'Tight':
            opt = 'Int=UltraFine Opt=(Tight,TS,CalcFC,NoEigen,MaxCycles=200) '
        elif self.opt_thresh == 'VeryTight':
            opt = 'Int=UltraFine Opt=(VeryTight,TS,CalcFC,NoEigen,MaxCycles=200) '
        else:
            opt = 'Opt=(TS,CalcFC,NoEigen,MaxCycles=200) '

        if self.temps[state] == 0.0 or self.pressures[state] == 0.0:
            freq = ''
        else:
            freq = 'Freq Temp={temp:.3f} Pressure={pressure:.5f}'.format(temp=self.temps[state], pressure=self.pressures[state])

        header = '''%NProcShared={n_proc:d}
#n {method:s} NoSymm SCF=XQC {opt:s}{freq:s}

 {label:s}

{charge:d} {mult:d}
'''.format(n_proc=self.n_proc, method=self.method, opt=opt, freq=freq, label=os.path.basename(optimizer), charge=self.charges[state], mult=self.mults[state])
        body = ''
        for j, (X, coord) in enumerate(zip(atoms, coords)):
            if j in self.frozen_atoms[state]:
                atom_type = -1
            else:
                atom_type = 0
            body += '{X:2s} {t:2d} {x:9f} {y:9f} {z:9f}\n'.format(X=X, t=atom_type, x=coord[0], y=coord[1], z=coord[2])
        footer = '\n'
        if self.gen_basis[state] != '':
            footer += self.gen_basis[state] + '\n\n'

        if not os.path.exists('{:s}.com'.format(optimizer)):
            f = open('{:s}.com'.format(optimizer), 'wt')
            f.write(header + body + footer)
            f.close()

        return

    def run(self, dry_run=False, verbose=False):
        n_struct = len(self.structures)
        for i in range(n_struct):
            sorted_optimizers = []
            for optimizer in self.optimizers[i]:
                if self.struc_types[i].upper() == 'TS':
                    output = self.run_ts_opt(optimizer, self.ts_criteria[i], dry_run, verbose)
                else:
                    output = self.run_geom_opt(optimizer, dry_run, verbose)
                if output is not None:
                    energy, cluster = output
                    self.energies[i].append(energy)
                    self.clusters[i].append(cluster)
                    sorted_optimizers.append(optimizer)
            self.optimizers[i] = sorted_optimizers
        if self.add_prefixes is not None:
            self.add_run()
        return

    def add_run(self):
        n_struct = len(self.structures)
        for i in range(n_struct):
            for optimizer in self.add_optimizers[i]:
                energy = 0.0
                if os.path.exists('{:s}.log'.format(optimizer)):
                    f = open('{:s}.log'.format(optimizer), 'rt')
                    for line in f:
                        if line.strip().startswith('BSSE energy'):
                            energy = float(line.split()[-1])
                    f.close()
                self.add_energies[i].append(energy)
        return

    def run_geom_opt(self, optimizer, dry_run=False, verbose=False):

        if not os.path.exists('{:s}.log'.format(optimizer)) or not check_normal_termination('{:s}.log'.format(optimizer)):
            if not dry_run:
                os.system('g16 {optimizer:s}.com > {optimizer:s}.log'.format(optimizer=optimizer))

        if os.path.exists('{:s}.log'.format(optimizer)):
            energies, clusters = read_geom_opt('{:s}.log'.format(optimizer))
            status = check_normal_termination('{:s}.log'.format(optimizer))
            if os.path.exists('{:s}.ilx'.format(optimizer)):
                f = open('{:s}.ilx'.format(optimizer), 'rt')
                code = f.read().strip()
                f.close()
                if code == 'keep':
                    status = -1
                elif code == 'drop':
                    status = -2
            if status == 0:
                return energies[-1], clusters[-1]
            elif status == 1:
                print(optimizer, 'Error termination')
                return
            elif status == 2:
                print(optimizer, 'Incomplete')
                return
            elif status == -1:
                if verbose:
                    print(optimizer, 'Override keep')
                return energies[-1], clusters[-1]
            elif status == -2:
                if verbose:
                    print(optimizer, 'Override drop')
                return
            else:
                print(optimizer, 'Unknown error')
                return
        else:
            print(optimizer, 'No output')
            return

    def run_ts_opt(self, optimizer, criteria, dry_run=False, verbose=False):

        if not os.path.exists('{:s}.log'.format(optimizer)) or not check_normal_termination('{:s}.log'.format(optimizer)):
            if not dry_run:
                os.system('g16 {optimizer:s}.com > {optimizer:s}.log'.format(optimizer=optimizer))

        if os.path.exists('{:s}.log'.format(optimizer)):
            energies, clusters = read_geom_opt('{:s}.log'.format(optimizer))
            status = check_normal_termination('{:s}.log'.format(optimizer))
            if status == 0 and not check_geometry(clusters[-1], criteria):
                status = 3
            if os.path.exists('{:s}.ilx'.format(optimizer)):
                f = open('{:s}.ilx'.format(optimizer), 'rt')
                code = f.read().strip()
                f.close()
                if code == 'keep':
                    status = -1
                elif code == 'drop':
                    status = -2
            if status == 0:
                return energies[-1], clusters[-1]
            elif status == 1:
                print(optimizer, 'Error termination')
                return
            elif status == 2:
                print(optimizer, 'Incomplete')
                return
            elif status == 3:
                print(optimizer, 'Wrong transition state')
                return
            elif status == -1:
                if verbose:
                    print(optimizer, 'Override keep')
                return energies[-1], clusters[-1]
            elif status == -2:
                if verbose:
                    print(optimizer, 'Override drop')
                return
            else:
                print(optimizer, 'Unknown error')
                return
        else:
            print(optimizer, 'No output')
            return

        return

    def get_orbitals(self):

        n_struct = len(self.structures)

        self.orbitals = [[] for i in range(n_struct)]
        for i in range(n_struct):
            for optimizer in self.optimizers[i]:
                orbitals = read_orbitals('{:s}.log'.format(optimizer))
                self.orbitals[i].append(orbitals[-1])

        return self.orbitals

    def get_thermochem(self, temps=None, pressures=None, vib_cutoff=None):

        n_struct = len(self.structures)

        if temps is None:
            temps = self.temps
        elif isinstance(temps, float):
            temps = [temps] * n_struct
        else:
            temps = temps

        if pressures is None:
            pressures = self.pressures
        elif isinstance(pressures, float):
            pressures = [pressures] * n_struct
        else:
            pressures = pressures

        if vib_cutoff is None:
            vib_cutoff = self.vib_cutoff

        self.enthalpies = [[] for i in range(n_struct)]
        self.entropies = [[] for i in range(n_struct)]
        self.gibbs_energies = [[] for i in range(n_struct)]
        for i in range(n_struct):
            if self.frozen_atoms[i] == []:
                elec = True
                trans = True
                rot = True
                vib = True
            else:
                elec = True
                trans = False
                rot = False
                vib = True
            for optimizer in self.optimizers[i]:
                E_e, H, S, G = read_thermochem('{:s}.log'.format(optimizer), temp=temps[i], pressure=pressures[i], vib_cutoff=vib_cutoff, elec=elec, trans=trans, rot=rot, vib=vib, verbose=True)
                self.enthalpies[i].append(H)
                self.entropies[i].append(S)
                self.gibbs_energies[i].append(G)

        return

    def get_gibbs_energies(self, temps=None, pressures=None, vib_cutoff=None):

        n_struct = len(self.structures)

        if temps is None:
            temps = self.temps
        if pressures is None:
            pressures = self.pressures
        if vib_cutoff is None:
            vib_cutoff = self.vib_cutoff

        if self.gibbs_energies == [[] for i in range(n_struct)] or temps != self.temps or pressures != self.pressures or vib_cutoff != self.vib_cutoff:
            self.get_thermochem(temps, pressures, vib_cutoff)

        return self.gibbs_energies

    def get_enthalpies(self, temps=None, pressures=None, vib_cutoff=None):

        n_struct = len(self.structures)

        if temps is None:
            temps = self.temps
        if pressures is None:
            pressures = self.pressures
        if vib_cutoff is None:
            vib_cutoff = self.vib_cutoff

        if self.enthalpies == [[] for i in range(n_struct)] or temps != self.temps or pressures != self.pressures or vib_cutoff != self.vib_cutoff:
            self.get_thermochem(temps, pressures, vib_cutoff)

        return self.enthalpies

    def get_entropies(self, temps=None, pressures=None, vib_cutoff=None):

        n_struct = len(self.structures)

        if temps is None:
            temps = self.temps
        if pressures is None:
            pressures = self.pressures
        if vib_cutoff is None:
            vib_cutoff = self.vib_cutoff

        if self.entropies == [[] for i in range(n_struct)] or temps != self.temps or pressures != self.pressures or vib_cutoff != self.vib_cutoff:
            self.get_thermochem(temps, pressures, vib_cutoff)

        return self.entropies

    def sort_conformers(self, e_window=None, r_thresh=None, exclude_atoms=None, exclude_elements=None, reorder=True):

        if e_window is None:
            e_window = self.e_window
        if r_thresh is None:
            r_thresh = self.r_thresh

        if exclude_atoms is None:
            exclude_atoms = self.exclude_atoms
        elif isinstance(exclude_atoms[0], int):
            exclude_atoms = [exclude_atoms] * n_struct

        if exclude_elements is None:
            exclude_elements = self.exclude_elements
        elif isinstance(exclude_elements, str):
            exclude_elements = [exclude_elements] * n_struct

        if self.add_prefixes is not None:
            self.add_sort_conformers(e_window, r_thresh, exclude_atoms, exclude_elements, reorder)

        n_struct = len(self.structures)
        for i in range(n_struct):
            sorted_degeneracies = []
            sorted_optimizers = []
            sorted_energies = []
            sorted_clusters = []
            if reorder:
                iterator = numpy.argsort(self.energies[i], kind='stable')
            else:
                iterator = [m for m, _ in enumerate(self.energies[i])]
            for m in iterator:
                if self.energies[i][m] > min(self.energies[i]) + e_window:
                    continue
                status = True
                for sorted_cluster in sorted_clusters:
                    atoms = self.clusters[i][m].get_chemical_symbols()
                    coords = self.clusters[i][m].get_positions()
                    sorted_coords = sorted_cluster.get_positions()
                    #match = MatchCoords(coords, sorted_coords)
                    #match.fit()
                    #coords = match.transform(coords)
                    indices = [j for j, X in enumerate(atoms) if j not in exclude_atoms[i] and X not in exclude_elements[i]]
                    if numpy.sqrt(numpy.mean((coords[indices]-sorted_coords[indices])**2)) < r_thresh:
                        status = False
                        break
                if not status:
                    continue
                sorted_degeneracies.append(self.degeneracies[i][m])
                sorted_optimizers.append(self.optimizers[i][m])
                sorted_energies.append(self.energies[i][m])
                sorted_clusters.append(self.clusters[i][m])
            self.degeneracies[i] = sorted_degeneracies
            self.optimizers[i] = sorted_optimizers
            self.energies[i] = sorted_energies
            self.clusters[i] = sorted_clusters

        if self.gibbs_energies != [[] for i in range(n_struct)]:
            self.get_thermochem()

        if self.orbitals != [[] for i in range(n_struct)]:
            self.get_orbitals()

        return

    def add_sort_conformers(self, e_window=None, r_thresh=None, exclude_atoms=None, exclude_elements=None, reorder=True):

        if e_window is None:
            e_window = self.e_window
        if r_thresh is None:
            r_thresh = self.r_thresh

        if exclude_atoms is None:
            exclude_atoms = self.exclude_atoms
        elif isinstance(exclude_atoms[0], int):
            exclude_atoms = [exclude_atoms] * n_struct

        if exclude_elements is None:
            exclude_elements = self.exclude_elements
        elif isinstance(exclude_elements, str):
            exclude_elements = [exclude_elements] * n_struct

        n_struct = len(self.structures)
        for i in range(n_struct):
            sorted_clusters = []
            add_sorted_optimizers = []
            add_sorted_energies = []
            if reorder:
                iterator = numpy.argsort(self.energies[i], kind='stable')
            else:
                iterator = [m for m, _ in enumerate(self.energies[i])]
            for m in iterator:
                if self.energies[i][m] > min(self.energies[i]) + e_window:
                    continue
                status = True
                for sorted_cluster in sorted_clusters:
                    atoms = self.clusters[i][m].get_chemical_symbols()
                    coords = self.clusters[i][m].get_positions()
                    sorted_coords = sorted_cluster.get_positions()
                    #match = MatchCoords(coords, sorted_coords)
                    #match.fit()
                    #coords = match.transform(coords)
                    indices = [j for j, X in enumerate(atoms) if j not in exclude_atoms[i] and X not in exclude_elements[i]]
                    if numpy.sqrt(numpy.mean((coords[indices]-sorted_coords[indices])**2)) < r_thresh:
                        status = False
                        break
                if not status:
                    continue
                add_sorted_optimizers.append(self.add_optimizers[i][m])
                add_sorted_energies.append(self.add_energies[i][m])
            self.add_optimizers[i] = add_sorted_optimizers
            self.add_energies[i] = add_sorted_energies

        return


import os

from ase import Atoms
from ase.io import read

from .gaussian_tools import check_normal_termination, check_geometry, read_geom_opt, read_thermochem


class Gaussian():

    def __init__(self, file_paths, prefixes,
            which=':', file_type='xyz',
            structure_types='EQ',
            charges=0, mults=4,
            temps=373.15, pressure=1.0, vib_cutoff=100.0,
            n_proc=24, method='wB97XD', basis='Gen',
            gen_basis='Cr 0\nDef2TZVP\n****\nSi O C H 0\nTZVP\n****\nF 0\n3-21G\n****',
            frozen_atoms=[0, 1, 2, 3],
            transition_state_criteria={(10, 11): (1.9, 2.4), (10, 13): (1.9, 2.4), (12, 13): (1.9, 2.4)}
            ):

        if isinstance(which, int):
            self.structures = [[read(file_path, which, file_type)] for file_path in file_paths]
        else:
            self.structures = [read(file_path, which, file_type) for file_path in file_paths]

        self.prefixes = prefixes

        n_struct = len(self.structures)

        if isinstance(structure_types, str):
            self.structure_types = [structure_types] * n_struct
        else:
            self.structure_types = structure_types

        if isinstance(charges, int):
            self.charges = [charges] * n_struct
        else:
            self.charges = charges

        if isinstance(mults, int):
            self.mults = [mults] * n_struct
        else:
            self.mults = mults

        if isinstance(temps, int):
            self.temps = [temps] * n_struct
        else:
            self.temps = temps

        self.pressure = pressure
        self.vib_cutoff = vib_cutoff

        self.n_proc = n_proc
        self.method = method
        self.basis = basis

        if isinstance(gen_basis, str):
            self.gen_basis = [gen_basis] * n_struct
        else:
            self.gen_basis = gen_basis

        if isinstance(frozen_atoms[0], int):
            self.frozen_atoms = [frozen_atoms] * n_struct
        else:
            self.frozen_atoms = frozen_atoms

        if isinstance(transition_state_criteria, dict):
            self.transition_state_criteria = [transition_state_criteria if self.structure_types[i].upper() == 'TS' else {} for i in range(n_struct)]
        else:
            self.transition_state_criteria = transition_state_criteria

        self.optimizers = [[] for i in range(n_struct)]
        self.energies = [[] for i in range(n_struct)]
        self.clusters = [[] for i in range(n_struct)]
        self.enthalpies = [[] for i in range(n_struct)]
        self.entropies = [[] for i in range(n_struct)]
        self.gibbs_energies = [[] for i in range(n_struct)]

        return

    def setup(self):
        n_struct = len(self.structures)
        for i in range(n_struct):
            n_digits = str(len(str(len(self.structures[i]))))
            for j, _ in enumerate(self.structures[i]):
                optimizer = ('{:s}.{:0' + n_digits + 'd}').format(self.prefixes[i], j)
                if self.structure_types[i].upper() == 'TS':
                    self.setup_transition_state_optimization(i, optimizer, self.structures[i][j])
                else:
                    self.setup_geometry_optimization(i, optimizer, self.structures[i][j])
                if optimizer not in self.optimizers[i]:
                    self.optimizers[i].append(optimizer)
        return self.optimizers

    def setup_geometry_optimization(self, state, optimizer, cluster):

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()

        header = '''%NProcShared={n_proc:d}
#n {method:s}/{basis:s} NoSymm SCF=XQC Opt=(MaxCycles=200) Freq Temp={temp:.3f} Pressure={pressure:.5f}

 {label:s}

{charge:d} {mult:d}
'''.format(n_proc=self.n_proc, method=self.method, basis=self.basis, label=os.path.basename(optimizer), charge=self.charges[state], mult=self.mults[state], temp=self.temps[state], pressure=self.pressure)
        body = ''
        for j, (X, coord) in enumerate(zip(atoms, coords)):
            if j in self.frozen_atoms[state]:
                atom_type = -1
            else:
                atom_type = 0
            body += '{X:2s} {t:2d} {x:9f} {y:9f} {z:9f}\n'.format(X=X, t=atom_type, x=coord[0], y=coord[1], z=coord[2])
        footer = '\n'
        if self.basis.upper() in ['GEN', 'GENECP']:
            footer += self.gen_basis[state] + '\n\n'

        if not os.path.exists('{:s}.com'.format(optimizer)):
            f = open('{:s}.com'.format(optimizer), 'wt')
            f.write(header + body + footer)
            f.close()

        return

    def setup_transition_state_optimization(self, state, optimizer, cluster):

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()

        header = '''%NProcShared={n_proc:d}
#n {method:s}/{basis:s} NoSymm SCF=XQC Opt=(TS,CalcFC,NoEigen,MaxCycles=200) Freq Temp={temp:.3f} Pressure={pressure:.5f}

 {label:s}

{charge:d} {mult:d}
'''.format(n_proc=self.n_proc, method=self.method, basis=self.basis, label=os.path.basename(optimizer), charge=self.charges[state], mult=self.mults[state], temp=self.temps[state], pressure=self.pressure)
        body = ''
        for j, (X, coord) in enumerate(zip(atoms, coords)):
            if j in self.frozen_atoms[state]:
                atom_type = -1
            else:
                atom_type = 0
            body += '{X:2s} {t:2d} {x:9f} {y:9f} {z:9f}\n'.format(X=X, t=atom_type, x=coord[0], y=coord[1], z=coord[2])
        footer = '\n'
        if self.basis.upper() in ['GEN', 'GENECP']:
            footer += self.gen_basis[state] + '\n\n'

        if not os.path.exists('{:s}.com'.format(optimizer)):
            f = open('{:s}.com'.format(optimizer), 'wt')
            f.write(header + body + footer)
            f.close()

        return

    def run(self, dry_run=False):
        n_struct = len(self.structures)
        for i in range(n_struct):
            for optimizer in self.optimizers[i]:
                if self.structure_types[i].upper() == 'TS':
                    output = self.run_transition_state_optimization(optimizer, self.transition_state_criteria[i], dry_run)
                else:
                    output = self.run_geometry_optimization(optimizer, dry_run)
                if output is not None:
                    energy, cluster = output
                    self.energies[i].append(energy)
                    self.clusters[i].append(cluster)
        return

    def run_geometry_optimization(self, optimizer, dry_run=False):

        if not os.path.exists('{:s}.log'.format(optimizer)) or not check_normal_termination('{:s}.log'.format(optimizer)):
            if not dry_run:
                os.system('g16 {optimizer:s}.com > {optimizer:s}.log'.format(optimizer=optimizer))

        if os.path.exists('{:s}.log'.format(optimizer)):
            status = check_normal_termination('{:s}.log'.format(optimizer))
            if status == True:
                energies, clusters = read_geom_opt('{:s}.log'.format(optimizer))
                return energies[-1], clusters[-1]
            elif status == False:
                print(optimizer, 'Error termination')
                return
            else:
                print(optimizer, 'Incomplete')
                return
        else:
            print(optimizer, 'No output')
            return

    def run_transition_state_optimization(self, optimizer, criteria, dry_run=False):

        if not os.path.exists('{:s}.log'.format(optimizer)) or not check_normal_termination('{:s}.log'.format(optimizer)):
            if not dry_run:
                os.system('g16 {optimizer:s}.com > {optimizer:s}.log'.format(optimizer=optimizer))

        if os.path.exists('{:s}.log'.format(optimizer)):
            status = check_normal_termination('{:s}.log'.format(optimizer))
            if status == True:
                energies, clusters = read_geom_opt('{:s}.log'.format(optimizer))
                if check_geometry(clusters[-1], criteria):
                    return energies[-1], clusters[-1]
                else:
                    print(optimizer, 'Wrong transition state')
                    return
            elif status == False:
                print(optimizer, 'Error termination')
                return
            else:
                print(optimizer, 'Incomplete')
                return
        else:
            print(optimizer, 'No output')
            return

        return

    def get_thermodynamics(self, temps=None, pressure=None, vib_cutoff=None):

        n_struct = len(self.structures)

        if temps is None:
            temps = self.temps
        elif isinstance(temps, int):
            temps = [temps] * n_struct

        if pressure is None:
            pressure = self.pressure
        elif pressure != self.pressure:
            self.pressure = pressure

        if vib_cutoff is None:
            vib_cutoff = self.vib_cutoff
        elif vib_cutoff != self.vib_cutoff:
            self.vib_cutoff = vib_cutoff

        self.gibbs_energies = [[] for i in range(n_struct)]
        self.enthalpies = [[] for i in range(n_struct)]
        self.entropies = [[] for i in range(n_struct)]
        for i in range(n_struct):
            for optimizer in self.optimizers[i]:
                E_e, H, S, G = read_thermochem('{:s}.log'.format(optimizer), temp=temps[i], pressure=pressure, vib_cutoff=vib_cutoff, verbose=True)
                self.gibbs_energies[i].append(G)
                self.enthalpies[i].append(H)
                self.entropies[i].append(S)

        return


    def get_gibbs_energies(self, temps=None, pressure=None, vib_cutoff=None):

        if temps is None:
            temps = self.temps
        if pressure is None:
            pressure = self.pressure
        if vib_cutoff is None:
            vib_cutoff = self.vib_cutoff

        n_struct = len(self.structures)
        if self.gibbs_energies == [[] for i in range(n_struct)]:
            self.get_thermodynamics(temps, pressure, vib_cutoff)

        if temps != self.temps or pressure != self.pressure or vib_cutoff != self.vib_cutoff:
            self.get_thermodynamics(temps, pressure, vib_cutoff)

        return self.gibbs_energies

    def get_enthalpies(self, temps=None, pressure=None, vib_cutoff=None):

        if temps is None:
            temps = self.temps
        if pressure is None:
            pressure = self.pressure
        if vib_cutoff is None:
            vib_cutoff = self.vib_cutoff

        n_struct = len(self.structures)
        if self.enthalpies == [[] for i in range(n_struct)]:
            self.get_thermodynamics(temps, pressure, vib_cutoff)

        if temps != self.temps or pressure != self.pressure or vib_cutoff != self.vib_cutoff:
            self.get_thermodynamics(temps, pressure, vib_cutoff)

        return self.enthalpies

    def get_entropies(self, temps=None, pressure=None, vib_cutoff=None):

        if temps is None:
            temps = self.temps
        if pressure is None:
            pressure = self.pressure
        if vib_cutoff is None:
            vib_cutoff = self.vib_cutoff

        n_struct = len(self.structures)
        if self.entropies == [[] for i in range(n_struct)]:
            self.get_thermodynamics(temps, pressure, vib_cutoff)

        if temps != self.temps or pressure != self.pressure or vib_cutoff != self.vib_cutoff:
            self.get_thermodynamics(temps, pressure, vib_cutoff)

        return entropies


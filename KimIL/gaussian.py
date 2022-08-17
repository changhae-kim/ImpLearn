import numpy
import os

from ase import Atoms
from ase.io import read

from .gaussian_tools import check_normal_termination, read_geometry_optimization, read_thermochemistry, check_geometry


class Gaussian():

    def __init__(self, catalyst_file_paths, reactant_file_paths, product_file_paths, transition_state_file_paths, prefix,
            file_type='xyz',
            charges=[0, 0, 0, 0], mults=[4, 4, 4, 4],
            temp=373.15, pressure=1.0,
            n_proc=24, method='wB97XD', basis='Gen',
            gen_basis='Cr 0\nDef2TZVP\n****\nSi O C H 0\nTZVP\n****',
            frozen_atoms=[6, 7, 8, 9, 10, 11, 12, 13],
            scan_params='B 16 17 S 10 0.1',
            scan_reverse=True,
            transition_state_criteria={(0, 14): (1.9, 2.4), (0, 16): (1.9, 2.4), (15, 16): (1.9, 2.4)}
            ):

        self.catalysts = [self.load_cluster(file_path, file_type) for file_path in catalyst_file_paths]
        self.reactants = [self.load_cluster(file_path, file_type) for file_path in reactant_file_paths]
        self.products = [self.load_cluster(file_path, file_type) for file_path in product_file_paths]
        self.transition_states = [self.load_cluster(file_path, file_type) for file_path in transition_state_file_paths]
        self.prefix = prefix
        self.charges = charges
        self.mults = mults
        self.temp = temp
        self.pressure = pressure

        self.set_parameters(n_proc, method, basis, gen_basis, frozen_atoms, scan_params, scan_reverse, transition_state_criteria)

        self.catalyst_optimizations = []
        self.reactant_optimizations = []
        self.product_optimizations = []
        self.scans = []
        self.transition_state_optimizations = []

        self.catalyst_energies = []
        self.reactant_energies = []
        self.product_energies = []
        self.scan_energies = []
        self.transition_state_energies = []

        self.catalyst_gibbs_energies = []
        self.reactant_gibbs_energies = []
        self.product_gibbs_energies = []
        self.transition_state_gibbs_energies = []

        self.catalyst_clusters = []
        self.reactant_clusters = []
        self.product_clusters = []
        self.scan_clusters = []
        self.transition_state_clusters = []

        return

    def load_cluster(self, file_path, file_type):
        cluster = read(file_path, 0, file_type)
        return cluster

    def set_parameters(self, n_proc=None, method=None, basis=None, gen_basis=None, frozen_atoms=None, scan_params=None, scan_reverse=None, transition_state_criteria=None):
        if n_proc is not None:
            self.n_proc = n_proc
        if method is not None:
            self.method = method
        if basis is not None:
            self.basis = basis
        if basis in ['gen', 'Gen', 'GEN'] and gen_basis is not None:
            self.gen_basis = gen_basis
        if frozen_atoms is not None:
            self.frozen_atoms = frozen_atoms
        if scan_params is not None:
            self.scan_params = scan_params
        if scan_reverse is not None:
            self.scan_reverse = scan_reverse
        if transition_state_criteria is not None:
            self.transition_state_criteria = transition_state_criteria
        return

    def setup(self, prefix=None):

        if prefix is None:
            prefix = self.prefix

        if self.product_optimizations == []:
            for i, cluster in enumerate(self.catalysts):
                label = '{:s}_b{:d}'.format(prefix, i)
                self.catalyst_optimizations.append(label)
                self.setup_geometry_optimization(label, cluster, self.charges[0], self.mults[0])

        if self.reactant_optimizations == []:
            for i, cluster in enumerate(self.reactants):
                label = '{:s}_r{:d}'.format(prefix, i)
                self.reactant_optimizations.append(label)
                self.setup_geometry_optimization(label, cluster, self.charges[1], self.mults[1])

        if self.product_optimizations == []:
            for i, cluster in enumerate(self.products):
                label = '{:s}_p{:d}'.format(prefix, i)
                self.product_optimizations.append(label)
                self.setup_geometry_optimization(label, cluster, self.charges[2], self.mults[2])

        if self.transition_states == []:

            if self.scan_energies != [] and self.transition_state_optimizations == []:
                for i, (scan_energies, scan_clusters) in enumerate(zip(self.scan_energies, self.scan_clusters)):
                    label = '{:s}_t{:d}'.format(prefix, i)
                    self.transition_state_optimizations.append(label)
                    self.setup_transition_state_optimization(label, scan_clusters[numpy.argmax(scan_energies)], self.charges[3], self.mults[3])

            elif self.scan_reverse and self.product_energies != [] and self.scans == []:
                for i, cluster in enumerate(product_clusters):
                    label = '{:s}_s{:d}'.format(prefix, i)
                    self.scans.append(label)
                    self.setup_scan(label, cluster, self.charges[3], self.mults[3])

            elif not self.scan_reverse and self.reactant_energies != [] and self.scans == []:
                for i, cluster in enumerate(reactant_clusters):
                    label = '{:s}_s{:d}'.format(prefix, i)
                    self.scans.append(label)
                    self.setup_scan(label, cluster, self.charges[3], self.mults[3])

        else:

            if self.transition_state_optimizations == []:
                for i, cluster in enumerate(self.transition_states):
                    label = '{:s}_t{:d}'.format(prefix, i)
                    self.transition_state_optimizations.append(label)
                    self.setup_transition_state_optimization(label, cluster, self.charges[3], self.mults[3])

        return

    def setup_geometry_optimization(self, label, cluster, charge, mult):

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()

        header = '''%NProcShared={n_proc:d}
#n {method:s}/{basis:s} NoSymm SCF=XQC Opt=(MaxCycles=200) Freq Temp={temp:.3f} Pressure={pressure:.5f}

 {label:s}

{charge:d} {mult:d}
'''.format(n_proc=self.n_proc, method=self.method, basis=self.basis, label=label, charge=charge, mult=mult, temp=self.temp, pressure=self.pressure)
        body = ''
        for j, (X, coord) in enumerate(zip(atoms, coords)):
            if j in self.frozen_atoms:
                atom_type = -1
            else:
                atom_type = 0
            body += '{X:s} {t:d} {x:f} {y:f} {z:f}\n'.format(X=X, t=atom_type, x=coord[0], y=coord[1], z=coord[2])
        footer = '\n'
        if self.basis in ['gen', 'Gen', 'GEN']:
            footer += self.gen_basis + '\n\n'

        if not os.path.exists('{:s}.com'.format(label)):
            f = open('{:s}.com'.format(label), 'wt')
            f.write(header + body + footer)
            f.close()

        return

    def setup_scan(self, label, cluster, charge, mult):

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()

        header = '''%NProcShared={n_proc:d}
#n PBEPBE/3-21G NoSymm SCF=XQC Opt=(ModRedundant,Loose,MaxCycles=200)

 {label:s}

{charge:d} {mult:d}
'''.format(n_proc=self.n_proc, label=label, charge=charge, mult=mult)
        body = ''
        for j, (X, coord) in enumerate(zip(atoms, coords)):
            if j in self.frozen_atoms:
                atom_type = -1
            else:
                atom_type = 0
            body += '{X:s} {t:d} {x:f} {y:f} {z:f}\n'.format(X=X, t=atom_type, x=coord[0], y=coord[1], z=coord[2])
        footer = '\n{:s}\n\n'.format(self.scan_params)

        if not os.path.exists('{:s}.com'.format(label)):
            f = open('{:s}.com'.format(label), 'wt')
            f.write(header + body + footer)
            f.close()

        return

    def setup_transition_state_optimization(self, label, cluster, charge, mult):

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()

        header = '''%NProcShared={n_proc:d}
#n {method:s}/{basis:s} NoSymm SCF=XQC Opt=(TS,NoEigen,CalcFC,MaxCycles=200) Freq Temp={temp:.3f} Pressure={pressure:.5f}

 {label:s}

{charge:d} {mult:d}
'''.format(n_proc=self.n_proc, method=self.method, basis=self.basis, label=label, charge=charge, mult=mult, temp=self.temp, pressure=self.pressure)
        body = ''
        for j, (X, coord) in enumerate(zip(atoms, coords)):
            if j in self.frozen_atoms:
                atom_type = -1
            else:
                atom_type = 0
            body += '{X:s} {t:d} {x:f} {y:f} {z:f}\n'.format(X=X, t=atom_type, x=coord[0], y=coord[1], z=coord[2])
        footer = '\n'
        if self.basis in ['gen', 'Gen', 'GEN']:
            footer += self.gen_basis + '\n\n'

        if not os.path.exists('{:s}.com'.format(label)):
            f = open('{:s}.com'.format(label), 'wt')
            f.write(header + body + footer)
            f.close()

        return

    def run(self, prefix=None, dry_run=False):

        if prefix is None:
            prefix = self.prefix

        if self.catalyst_energies == []:
            for label in self.catalyst_optimizations:
                output = self.run_geometry_optimization(label, dry_run)
                if output is not None:
                    optimized_energy, optimized_cluster = output
                    self.catalyst_energies.append(optimized_energy)
                    self.catalyst_clusters.append(optimized_cluster)

        if self.reactant_energies == []:
            for label in self.reactant_optimizations:
                output = self.run_geometry_optimization(label, dry_run)
                if output is not None:
                    optimized_energy, optimized_cluster = output
                    self.reactant_energies.append(optimized_energy)
                    self.reactant_clusters.append(optimized_cluster)

        if self.product_energies == []:
            for label in self.product_optimizations:
                output = self.run_geometry_optimization(label, dry_run)
                if output is not None:
                    optimized_energy, optimized_cluster = output
                    self.product_energies.append(optimized_energy)
                    self.product_clusters.append(optimized_cluster)

        if self.scan_energies == []:
            for label in self.scans:
                output = self.run_scan(label, dry_run)
                if output is not None:
                    scan_energies, scan_clusters = output
                    self.scan_energies.append(scan_energies)
                    self.scan_clusters.append(scan_clusters)

        if self.transition_state_energies == []:
            for label in self.transition_state_optimizations:
                output = self.run_transition_state_optimization(label, dry_run)
                if output is not None:
                    optimized_energy, optimized_cluster = output
                    self.transition_state_energies.append(optimized_energy)
                    self.transition_state_clusters.append(optimized_cluster)

        return

    def run_geometry_optimization(self, label, dry_run=False):

        if not os.path.exists('{:s}.log'.format(label)) or not check_normal_termination('{:s}.log'.format(label)):
            if not dry_run:
                os.system('g16 {label:s}.com > {label:s}.log'.format(label=label))

        if os.path.exists('{:s}.log'.format(label)):
            if check_normal_termination('{:s}.log'.format(label)):
                energies, clusters = read_geometry_optimization('{:s}.log'.format(label))
                return energies[-1], clusters[-1]
            else:
                print(label, 'Error termination')
                return
        else:
            print(label, 'No output')
            return

    def run_scan(self, label, dry_run=False):

        if not os.path.exists('{:s}.log'.format(label)):
            if not dry_run:
                os.system('g16 {label:s}.com > {label:s}.log'.format(label=label))

        if os.path.exists('{:s}.log'.format(label)):
            energies, clusters = read_geometry_optimization('{:s}.log'.format(label))
            return energies, clusters
        else:
            print(label, 'No output')
            return

    def run_transition_state_optimization(self, label, dry_run=False):

        if not os.path.exists('{:s}.log'.format(label)) or not check_normal_termination('{:s}.log'.format(label)):
            if not dry_run:
                os.system('g16 {label:s}.com > {label:s}.log'.format(label=label))

        if os.path.exists('{:s}.log'.format(label)):
            if check_normal_termination('{:s}.log'.format(label)):
                energies, clusters = read_geometry_optimization('{:s}.log'.format(label))
                if check_geometry(clusters[-1], self.transition_state_criteria):
                    return energies[-1], clusters[-1]
                else:
                    print(label, 'Wrong transition state')
                    return
            else:
                print(label, 'Error termination')
                return
        else:
            print(label, 'No output')
            return

        return

    def get_gibbs_energies(self, prefix=None, temp=None, pressure=None):

        if prefix is None:
            prefix = self.prefix
        if temp is None:
            temp = self.temp
        if pressure is None:
            pressure = self.pressure

        if self.catalyst_gibbs_energies == []:
            for label in self.catalyst_optimizations:
                gibbs_energy = read_thermochemistry('{:s}.log'.format(label), temp=self.temp, pressure=self.pressure)
                self.catalyst_gibbs_energies.append(gibbs_energy)

        if self.reactant_gibbs_energies == []:
            for label in self.reactant_optimizations:
                gibbs_energy = read_thermochemistry('{:s}.log'.format(label), temp=self.temp, pressure=self.pressure)
                self.reactant_gibbs_energies.append(gibbs_energy)

        if self.product_gibbs_energies == []:
            for label in self.product_optimizations:
                gibbs_energy = read_thermochemistry('{:s}.log'.format(label), temp=self.temp, pressure=self.pressure)
                self.product_gibbs_energies.append(gibbs_energy)

        if self.transition_state_gibbs_energies == []:
            for label in self.transition_state_optimizations:
                gibbs_energy = read_thermochemistry('{:s}.log'.format(label), temp=self.temp, pressure=self.pressure)
                self.transition_state_gibbs_energies.append(gibbs_energy)

        if temp != self.temp or pressure != self.pressure:

            catalyst_gibbs_energies = []
            for label in self.catalyst_optimizations:
                gibbs_energy = read_thermochemistry('{:s}.log'.format(label), temp=self.temp, pressure=self.pressure)
                catalyst_gibbs_energies.append(gibbs_energy)

            reactant_gibbs_energies = []
            for label in self.reactant_optimizations:
                gibbs_energy = read_thermochemistry('{:s}.log'.format(label), temp=self.temp, pressure=self.pressure)
                reactant_gibbs_energies.append(gibbs_energy)

            product_gibbs_energies = []
            for label in self.product_optimizations:
                gibbs_energy = read_thermochemistry('{:s}.log'.format(label), temp=self.temp, pressure=self.pressure)
                product_gibbs_energies.append(gibbs_energy)

            transition_state_gibbs_energies = []
            for label in self.transition_state_optimizations:
                gibbs_energy = read_thermochemistry('{:s}.log'.format(label), temp=self.temp, pressure=self.pressure)
                transition_state_gibbs_energies.append(gibbs_energy)

        else:

            catalyst_gibbs_energies = self.catalyst_gibbs_energies
            reactant_gibbs_energies = self.reactant_gibbs_energies
            product_gibbs_energies = self.product_gibbs_energies
            transition_state_gibbs_energies = self.transition_state_gibbs_energies

        return catalyst_gibbs_energies, reactant_gibbs_energies, product_gibbs_energies, transition_state_gibbs_energies


if __name__ == '__main__':


    gauss = Gaussian(
            ['tests/A_0001_L-butyl.xyz', 'tests/A_0001_R-butyl.xyz'],
            ['tests/A_0001_L-butyl-R-ethylene.xyz', 'tests/A_0001_R-butyl-L-ethylene.xyz'],
            ['tests/A_0001_R-hexyl.xyz', 'tests/A_0001_L-hexyl.xyz'],
            ['tests/A_0001_LR-transition-state.xyz', 'tests/A_0001_RL-transition-state.xyz'],
            'A_0001')
    gauss.setup()
    gauss.run()
    gibbs_energies = gauss.get_gibbs_energies()
    print(gibbs_energies)


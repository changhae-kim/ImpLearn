import numpy
import os

from ase import Atoms
from ase.io import read

from gaussian_tools import check_normal_termination, read_geometry_optimization, read_thermochemistry, check_geometry


class Gaussian():

    def __init__(self, catalyst_file_paths, reactant_file_paths, product_file_paths, transition_file_paths, prefix,
            file_type='xyz',
            charges=[0, 0, 0, 0], mults=[4, 4, 4, 4],
            n_proc=24, method='wB97XD', basis='Gen',
            gen_basis='Cr 0\nDef2TZVP\n****\nSi O C H 0\nTZVP\n****',
            frozen_atoms=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            scan_params='B 19 20 S 10 0.1',
            scan_reverse=True,
            transition_criteria={(0, 17): (1.9, 2.4), (0, 19): (1.9, 2.4), (18, 19): (1.9, 2.4)}
            ):

        self.catalysts = [self.load_cluster(file_path, file_type) for file_path in catalyst_file_paths]
        self.reactants = [self.load_cluster(file_path, file_type) for file_path in reactant_file_paths]
        self.products = [self.load_cluster(file_path, file_type) for file_path in product_file_paths]
        self.transitions = [self.load_cluster(file_path, file_type) for file_path in transition_file_paths]
        self.prefix = prefix
        self.charges = charges
        self.mults = mults

        self.set_parameters(n_proc, method, basis, gen_basis, frozen_atoms, scan_params, scan_reverse, transition_criteria)

        self.catalyst_optimizations = []
        self.reactant_optimizations = []
        self.product_optimizations = []
        self.scans = []
        self.transition_optimizations = []

        self.catalyst_energies = []
        self.reactant_energies = []
        self.product_energies = []
        self.scan_energies = []
        self.transition_energies = []

        self.catalyst_gibbs_energies = []
        self.reactant_gibbs_energies = []
        self.product_gibbs_energies = []
        self.transition_gibbs_energies = []

        self.catalyst_clusters = []
        self.reactant_clusters = []
        self.product_clusters = []
        self.scan_clusters = []
        self.transition_clusters = []

        return

    def load_cluster(self, file_path, file_type):
        cluster = read(file_path, 0, file_type)
        return cluster

    def set_parameters(self, n_proc=None, method=None, basis=None, gen_basis=None, frozen_atoms=None, scan_params=None, scan_reverse=None, transition_criteria=None):
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
        if transition_criteria is not None:
            self.transition_criteria = transition_criteria
        return

    def setup(self, prefix=None):

        if prefix is None:
            prefix = self.prefix

        if self.product_optimizations == []:
            for i, cluster in enumerate(self.catalysts):
                label = '{:s}_B_{:d}'.format(prefix, i)
                self.catalyst_optimizations.append(label)
                self.setup_geometry_optimization(label, cluster, self.charges[0], self.mults[0])

        if self.reactant_optimizations == []:
            for i, cluster in enumerate(self.reactants):
                label = '{:s}_R_{:d}'.format(prefix, i)
                self.reactant_optimizations.append(label)
                self.setup_geometry_optimization(label, cluster, self.charges[1], self.mults[1])

        if self.product_optimizations == []:
            for i, cluster in enumerate(self.products):
                label = '{:s}_P_{:d}'.format(prefix, i)
                self.product_optimizations.append(label)
                self.setup_geometry_optimization(label, cluster, self.charges[2], self.mults[2])

        if self.transitions == []:

            if self.scan_energies != [] and self.transition_optimizations == []:
                for i, (scan_energies, scan_clusters) in enumerate(zip(self.scan_energies, self.scan_clusters)):
                    label = '{:s}_T_{:d}'.format(prefix, i)
                    self.transition_optimizations.append(label)
                    self.setup_transition_optimization(label, scan_clusters[numpy.argmax(scan_energies)], self.charges[3], self.mults[3])

            elif self.scan_reverse and self.product_energies != [] and self.scans == []:
                for i, cluster in enumerate(product_clusters):
                    label = '{:s}_S_{:d}'.format(prefix, i)
                    self.scans.append(label)
                    self.setup_scan(label, cluster, self.charges[3], self.mults[3])

            elif not self.scan_reverse and self.reactant_energies != [] and self.scans == []:
                for i, cluster in enumerate(reactant_clusters):
                    label = '{:s}_S_{:d}'.format(prefix, i)
                    self.scans.append(label)
                    self.setup_scan(label, cluster, self.charges[3], self.mults[3])

        else:

            if self.transition_optimizations == []:
                for i, cluster in enumerate(self.transitions):
                    label = '{:s}_T_{:d}'.format(prefix, i)
                    self.transition_optimizations.append(label)
                    self.setup_transition_optimization(label, cluster, self.charges[3], self.mults[3])

        return

    def setup_geometry_optimization(self, label, cluster, charge, mult):

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()

        header = '''%NProcShared={n_proc:d}
#n {method:s}/{basis:s} NoSymm SCF=XQC Opt=(MaxCycles=200) Freq

 {label:s}

{charge:d} {mult:d}
'''.format(n_proc=self.n_proc, method=self.method, basis=self.basis, label=label, charge=charge, mult=mult)
        body = ''
        for j, (X, coord) in enumerate(zip(atoms, coords)):
            if j in self.frozen_atoms:
                frozen = -1
            else:
                frozen = 0
            body += '{X:s} {frozen:d} {x:f} {y:f} {z:f}\n'.format(X=X, frozen=frozen, x=coord[0], y=coord[1], z=coord[2])
        footer = '\n'
        if self.basis in ['gen', 'Gen', 'GEN']:
            footer += self.gen_basis + '\n\n'

        if os.path.exists('{:s}.com'.format(label)):
            print('setup_geometry_optimization(): {:s}.com already exists'.format(label))
        else:
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
                frozen = -1
            else:
                frozen = 0
            body += '{X:s} {frozen:d} {x:f} {y:f} {z:f}\n'.format(X=X, frozen=frozen, x=coord[0], y=coord[1], z=coord[2])
        footer = '\n{:s}\n\n'.format(self.scan_params)

        if os.path.exists('{:s}.com'.format(label)):
            print('setup_scan(): {:s}.com already exists'.format(label))
        else:
            f = open('{:s}.com'.format(label), 'wt')
            f.write(header + body + footer)
            f.close()

        return

    def setup_transition_optimization(self, label, cluster, charge, mult):

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()

        header = '''%NProcShared={n_proc:d}
#n {method:s}/{basis:s} NoSymm SCF=XQC Opt=(TS,NoEigen,CalcFC,MaxCycles=200) Freq

 {label:s}

{charge:d} {mult:d}
'''.format(n_proc=self.n_proc, method=self.method, basis=self.basis, label=label, charge=charge, mult=mult)
        body = ''
        for j, (X, coord) in enumerate(zip(atoms, coords)):
            if j in self.frozen_atoms:
                frozen = -1
            else:
                frozen = 0
            body += '{X:s} {frozen:d} {x:f} {y:f} {z:f}\n'.format(X=X, frozen=frozen, x=coord[0], y=coord[1], z=coord[2])
        footer = '\n'
        if self.basis in ['gen', 'Gen', 'GEN']:
            footer += self.gen_basis + '\n\n'

        if os.path.exists('{:s}.com'.format(label)):
            print('setup_transition_optimization(): {:s}.com already exists'.format(label))
        else:
            f = open('{:s}.com'.format(label), 'wt')
            f.write(header + body + footer)
            f.close()

        return

    def run(self, prefix=None):

        if prefix is None:
            prefix = self.prefix

        if self.catalyst_energies == []:
            for label in self.catalyst_optimizations:
                optimized_energy, gibbs_energy, optimized_cluster = self.run_optimization(label)
                self.catalyst_energies.append(optimized_energy)
                self.catalyst_gibbs_energies.append(gibbs_energy)
                self.catalyst_clusters.append(optimized_cluster)

        if self.reactant_energies == []:
            for label in self.reactant_optimizations:
                optimized_energy, gibbs_energy, optimized_cluster = self.run_optimization(label)
                self.reactant_energies.append(optimized_energy)
                self.reactant_gibbs_energies.append(gibbs_energy)
                self.reactant_clusters.append(optimized_cluster)

        if self.product_energies == []:
            for label in self.product_optimizations:
                optimized_energy, gibbs_energy, optimized_cluster = self.run_optimization(label)
                self.product_energies.append(optimized_energy)
                self.product_gibbs_energies.append(gibbs_energy)
                self.product_clusters.append(optimized_cluster)

        if self.scan_energies == []:
            for label in self.scans:
                scan_energies, scan_clusters = self.run_scan(label)
                self.scan_energies.append(scan_energies)
                self.scan_clusters.append(scan_clusters)

        if self.transition_energies == []:
            for label in self.transition_optimizations:
                optimized_energy, gibbs_energy, optimized_cluster = self.run_optimization(label)
                self.transition_energies.append(optimized_energy)
                self.transition_gibbs_energies.append(gibbs_energy)
                self.transition_clusters.append(optimized_cluster)

        return

    def run_optimization(self, label):

        if os.path.exists('{:s}.log'.format(label)):
            if check_normal_termination('{:s}.log'.format(label)):
                print('run_optimization(): {:s}.log already done'.format(label))
            else:
                os.system('g16 {label:s}.com > {label:s}.log'.format(label=label))
        else:
            os.system('g16 {label:s}.com > {label:s}.log'.format(label=label))

        if check_normal_termination('{:s}.log'.format(label)):
            energies, clusters = read_geometry_optimization('{:s}.log'.format(label))
            gibbs_energy = read_thermochemistry('{:s}.log'.format(label))
            if check_geometry(clusters[-1], self.transition_criteria):
                return energies[0], gibbs_energy, clusters[0]
            else:
                return
        else:
            return

    def run_scan(self, label):

        if os.path.exists('{:s}.log'.format(label)):
            print('run_scan(): {:s}.log already done'.format(label))
        else:
            os.system('g16 {label:s}.com > {label:s}.log'.format(label=label))

        energies, clusters = read_geometry_optimization('{:s}.log'.format(label))

        return energies, clusters

    def get_gibbs_energies(self):
        return self.catalyst_gibbs_energies, self.reactant_gibbs_energies, self.product_gibbs_energies, self.transition_gibbs_energies


if __name__ == '__main__':


    gauss = Gaussian(
            ['tests/A_0000_L_butyl.xyz', 'tests/A_0000_R_butyl.xyz'],
            ['tests/A_0000_L_butyl_R_ethylene.xyz', 'tests/A_0000_R_butyl_L_ethylene.xyz'],
            ['tests/A_0000_R_hexyl.xyz', 'tests/A_0000_L_hexyl.xyz'],
            ['tests/A_0000_LR_transition.xyz', 'tests/A_0000_RL_transition.xyz'],
            'A_0000')
    gauss.setup()
    gauss.run()



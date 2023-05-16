import numpy
import os

from ase import Atoms
from ase.io import read

from .gaussian_tools import check_normal_termination, read_geom_opt, read_thermochem, check_geometry


class Gaussian():

    def __init__(self, catalyst_file_paths, reactant_file_paths, product_file_paths, transition_state_file_paths, prefix,
            file_type='xyz',
            charges=0, mults=4,
            temp=373.15, pressure=1.0,
            n_proc=24, method='wB97XD', basis='Gen',
            gen_basis='Cr 0\nDef2TZVP\n****\nSi O C H 0\nTZVP\n****',
            preopt='wB97XD/3-21G',
            frozen_atoms=[6, 7, 8, 9, 10, 11, 12, 13],
            transition_state_criteria={(0, 14): (1.9, 2.4), (0, 16): (1.9, 2.4), (15, 16): (1.9, 2.4)},
            freq_cutoff=0.0
            ):

        self.catalysts = [self.load_cluster(file_path, file_type) for file_path in catalyst_file_paths]
        self.reactants = [self.load_cluster(file_path, file_type) for file_path in reactant_file_paths]
        self.products = [self.load_cluster(file_path, file_type) for file_path in product_file_paths]
        self.transition_states = [self.load_cluster(file_path, file_type) for file_path in transition_state_file_paths]
        self.prefix = prefix

        if isinstance(charges, int):
            self.charges = [charges, charges, charges, charges]
        else:
            self.charges = charges

        if isinstance(mults, int):
            self.mults = [mults, mults, mults, mults]
        else:
            self.mults = mults

        self.temp = temp
        self.pressure = pressure

        self.n_proc = None
        self.method = None
        self.basis = None
        self.gen_basis = None
        self.preopt = None
        self.frozen_atoms = None
        self.transition_state_criteria = None
        self.freq_cutoff = None

        self.set_parameters(n_proc, method, basis, gen_basis, preopt, frozen_atoms, transition_state_criteria, freq_cutoff)

        self.catalyst_optimizations = []
        self.reactant_optimizations = []
        self.product_optimizations = []
        self.transition_state_optimizations = []

        self.catalyst_energies = []
        self.reactant_energies = []
        self.product_energies = []
        self.transition_state_energies = []

        self.catalyst_enthalpies = []
        self.reactant_enthalpies = []
        self.product_enthalpies = []
        self.transition_state_enthalpies = []

        self.catalyst_entropies = []
        self.reactant_entropies = []
        self.product_entropies = []
        self.transition_state_entropies = []

        self.catalyst_gibbs_energies = []
        self.reactant_gibbs_energies = []
        self.product_gibbs_energies = []
        self.transition_state_gibbs_energies = []

        self.catalyst_clusters = []
        self.reactant_clusters = []
        self.product_clusters = []
        self.transition_state_clusters = []

        return

    def load_cluster(self, file_path, file_type):
        cluster = read(file_path, -1, file_type)
        return cluster

    def set_parameters(self, n_proc=None, method=None, basis=None, gen_basis=None, preopt=None, frozen_atoms=None, transition_state_criteria=None, freq_cutoff=None):
        if n_proc is not None:
            self.n_proc = n_proc
        if method is not None:
            self.method = method
        if basis is not None:
            self.basis = basis
        if gen_basis is not None:
            if isinstance(gen_basis, str):
                self.gen_basis = [gen_basis, gen_basis, gen_basis, gen_basis]
            else:
                self.gen_basis = gen_basis
        if preopt is not None:
            self.preopt = preopt
        if frozen_atoms is not None:
            if isinstance(frozen_atoms[0], int):
                self.frozen_atoms = [frozen_atoms, frozen_atoms, frozen_atoms, frozen_atoms]
            else:
                self.frozen_atoms = frozen_atoms
        if transition_state_criteria is not None:
            if isinstance(transition_state_criteria, dict):
                self.transition_state_criteria = [transition_state_criteria] * len(self.transition_states)
            else:
                self.transition_state_criteria = transition_state_criteria
        if freq_cutoff is not None:
            self.freq_cutoff = freq_cutoff
        return

    def setup(self, prefix=None):

        if prefix is None:
            prefix = self.prefix

        if self.catalyst_optimizations == []:
            for i, cluster in enumerate(self.catalysts):
                label = '{:s}_b{:d}'.format(prefix, i)
                self.catalyst_optimizations.append(label)
                self.setup_geometry_optimization(label, cluster, 0)

        if self.reactant_optimizations == []:
            for i, cluster in enumerate(self.reactants):
                label = '{:s}_r{:d}'.format(prefix, i)
                self.reactant_optimizations.append(label)
                self.setup_geometry_optimization(label, cluster, 1)

        if self.product_optimizations == []:
            for i, cluster in enumerate(self.products):
                label = '{:s}_p{:d}'.format(prefix, i)
                self.product_optimizations.append(label)
                self.setup_geometry_optimization(label, cluster, 2)

        if self.transition_state_optimizations == []:
            for i, cluster in enumerate(self.transition_states):
                label = '{:s}_t{:d}'.format(prefix, i)
                self.transition_state_optimizations.append(label)
                self.setup_transition_state_optimization(label, cluster)

        return

    def setup_geometry_optimization(self, label, cluster, state):

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()

        if self.preopt is not None:
            header = '''%NProcShared={n_proc:d}
%Chk = {basename:s}.chk
#n {preopt:s} NoSymm SCF=XQC Opt=(Loose,MaxCycles=200)

 {label:s}_preopt

{charge:d} {mult:d}
'''.format(n_proc=self.n_proc, basename=os.path.basename(label), preopt=self.preopt, label=label, charge=self.charges[state], mult=self.mults[state])
            body = ''
            for j, (X, coord) in enumerate(zip(atoms, coords)):
                if j in self.frozen_atoms[state]:
                    atom_type = -1
                else:
                    atom_type = 0
                body += '{X:2s} {t:2d} {x:9f} {y:9f} {z:9f}\n'.format(X=X, t=atom_type, x=coord[0], y=coord[1], z=coord[2])
            footer = '''
--Link1--
%NProcShared={n_proc:d}
%Chk = {basename:s}.chk
%NoSave
#n {method:s}/{basis:s} NoSymm Geom=(Check,AddGIC) SCF=XQC Opt=(MaxCycles=200) Freq Temp={temp:.3f} Pressure={pressure:.5f}

 {label:s}

{charge:d} {mult:d}

!

'''.format(n_proc=self.n_proc, basename=os.path.basename(label), method=self.method, basis=self.basis, label=label, charge=self.charges[state], mult=self.mults[state], temp=self.temp, pressure=self.pressure)
            if self.basis.upper() in ['GEN', 'GENECP']:
                footer += self.gen_basis[state] + '\n\n'

        else:
            header = '''%NProcShared={n_proc:d}
#n {method:s}/{basis:s} NoSymm SCF=XQC Opt=(MaxCycles=200) Freq Temp={temp:.3f} Pressure={pressure:.5f}

 {label:s}

{charge:d} {mult:d}
'''.format(n_proc=self.n_proc, method=self.method, basis=self.basis, label=label, charge=self.charges[state], mult=self.mults[state], temp=self.temp, pressure=self.pressure)
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

        if not os.path.exists('{:s}.com'.format(label)):
            f = open('{:s}.com'.format(label), 'wt')
            f.write(header + body + footer)
            f.close()

        return

    def setup_transition_state_optimization(self, label, cluster, state=3):

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()

        if self.preopt is not None:
            header = '''%NProcShared={n_proc:d}
%Chk = {basename:s}.chk
#n {preopt:s} NoSymm SCF=XQC Opt=(Loose,TS,CalcFC,NoEigen,MaxCycles=200)

 {label:s}_preopt

{charge:d} {mult:d}
'''.format(n_proc=self.n_proc, basename=os.path.basename(label), preopt=self.preopt, label=label, charge=self.charges[state], mult=self.mults[state])
            body = ''
            for j, (X, coord) in enumerate(zip(atoms, coords)):
                if j in self.frozen_atoms[state]:
                    atom_type = -1
                else:
                    atom_type = 0
                body += '{X:2s} {t:2d} {x:9f} {y:9f} {z:9f}\n'.format(X=X, t=atom_type, x=coord[0], y=coord[1], z=coord[2])
            footer = '''
--Link1--
%NProcShared={n_proc:d}
%Chk = {basename:s}.chk
%NoSave
#n {method:s}/{basis:s} NoSymm Geom=(Check,AddGIC) SCF=XQC Opt=(TS,CalcFC,NoEigen,MaxCycles=200) Freq Temp={temp:.3f} Pressure={pressure:.5f}

 {label:s}

{charge:d} {mult:d}

!

'''.format(n_proc=self.n_proc, basename=os.path.basename(label), method=self.method, basis=self.basis, label=label, charge=self.charges[state], mult=self.mults[state], temp=self.temp, pressure=self.pressure)
            if self.basis.upper() in ['GEN', 'GENECP']:
                footer += self.gen_basis[state] + '\n\n'

        else:
            header = '''%NProcShared={n_proc:d}
#n {method:s}/{basis:s} NoSymm SCF=XQC Opt=(TS,CalcFC,NoEigen,MaxCycles=200) Freq Temp={temp:.3f} Pressure={pressure:.5f}

 {label:s}

{charge:d} {mult:d}
'''.format(n_proc=self.n_proc, method=self.method, basis=self.basis, label=label, charge=self.charges[state], mult=self.mults[state], temp=self.temp, pressure=self.pressure)
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

        if not os.path.exists('{:s}.com'.format(label)):
            f = open('{:s}.com'.format(label), 'wt')
            f.write(header + body + footer)
            f.close()

        return

    def run(self, dry_run=False):

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

        if self.transition_state_energies == []:
            for label, criteria in zip(self.transition_state_optimizations, self.transition_state_criteria):
                output = self.run_transition_state_optimization(label, criteria, dry_run)
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
            status = check_normal_termination('{:s}.log'.format(label))
            if status == True:
                energies, clusters = read_geom_opt('{:s}.log'.format(label))
                return energies[-1], clusters[-1]
            elif status == False:
                print(label, 'Error termination')
                return
            else:
                print(label, 'Incomplete')
                return
        else:
            print(label, 'No output')
            return

    def run_transition_state_optimization(self, label, criteria, dry_run=False):

        if not os.path.exists('{:s}.log'.format(label)) or not check_normal_termination('{:s}.log'.format(label)):
            if not dry_run:
                os.system('g16 {label:s}.com > {label:s}.log'.format(label=label))

        if os.path.exists('{:s}.log'.format(label)):
            status = check_normal_termination('{:s}.log'.format(label))
            if status == True:
                energies, clusters = read_geom_opt('{:s}.log'.format(label))
                if check_geometry(clusters[-1], criteria):
                    return energies[-1], clusters[-1]
                else:
                    print(label, 'Wrong transition state')
                    return
            elif status == False:
                print(label, 'Error termination')
                return
            else:
                print(label, 'Incomplete')
                return
        else:
            print(label, 'No output')
            return

        return

    def get_thermodynamics(self, temp=None, pressure=None, freq_cutoff=None):

        if temp is None:
            temp = self.temp
        elif temp != self.temp:
            self.temp = temp

        if pressure is None:
            pressure = self.pressure
        elif pressure != self.pressure:
            self.pressure = pressure

        if freq_cutoff is None:
            freq_cutoff = self.freq_cutoff
        elif freq_cutoff != self.freq_cutoff:
            self.freq_cutoff = freq_cutoff

        self.catalyst_gibbs_energies = []
        self.catalyst_enthalpies = []
        self.catalyst_entropies = []
        for label in self.catalyst_optimizations:
            E_e, H, S, G = read_thermochem('{:s}.log'.format(label), temp=self.temp, pressure=self.pressure, freq_cutoff=self.freq_cutoff, verbose=True)
            self.catalyst_gibbs_energies.append(G)
            self.catalyst_enthalpies.append(H)
            self.catalyst_entropies.append(S)

        self.reactant_gibbs_energies = []
        self.reactant_enthalpies = []
        self.reactant_entropies = []
        for label in self.reactant_optimizations:
            E_e, H, S, G = read_thermochem('{:s}.log'.format(label), temp=self.temp, pressure=self.pressure, freq_cutoff=self.freq_cutoff, verbose=True)
            self.reactant_gibbs_energies.append(G)
            self.reactant_enthalpies.append(H)
            self.reactant_entropies.append(S)

        self.product_gibbs_energies = []
        self.product_enthalpies = []
        self.product_entropies = []
        for label in self.product_optimizations:
            E_e, H, S, G = read_thermochem('{:s}.log'.format(label), temp=self.temp, pressure=self.pressure, freq_cutoff=self.freq_cutoff, verbose=True)
            self.product_gibbs_energies.append(G)
            self.product_enthalpies.append(H)
            self.product_entropies.append(S)

        self.transition_state_gibbs_energies = []
        self.transition_state_enthalpies = []
        self.transition_state_entropies = []
        for label in self.transition_state_optimizations:
            E_e, H, S, G = read_thermochem('{:s}.log'.format(label), temp=self.temp, pressure=self.pressure, freq_cutoff=self.freq_cutoff, verbose=True)
            self.transition_state_gibbs_energies.append(G)
            self.transition_state_enthalpies.append(H)
            self.transition_state_entropies.append(S)

        return


    def get_gibbs_energies(self, temp=None, pressure=None, freq_cutoff=None):

        if temp is None:
            temp = self.temp
        if pressure is None:
            pressure = self.pressure
        if freq_cutoff is None:
            freq_cutoff = self.freq_cutoff

        if self.catalyst_gibbs_energies == [] or self.reactant_gibbs_energies == [] or self.product_gibbs_energies == [] or self.transition_state_gibbs_energies == []:
            self.get_thermodynamics(temp, pressure, freq_cutoff)

        if temp != self.temp or pressure != self.pressure or freq_cutoff != self.freq_cutoff:
            self.get_thermodynamics(temp, pressure, freq_cutoff)

        catalyst_gibbs_energies = self.catalyst_gibbs_energies
        reactant_gibbs_energies = self.reactant_gibbs_energies
        product_gibbs_energies = self.product_gibbs_energies
        transition_state_gibbs_energies = self.transition_state_gibbs_energies

        return catalyst_gibbs_energies, reactant_gibbs_energies, product_gibbs_energies, transition_state_gibbs_energies

    def get_enthalpies(self, temp=None, pressure=None, freq_cutoff=None):

        if temp is None:
            temp = self.temp
        if pressure is None:
            pressure = self.pressure
        if freq_cutoff is None:
            freq_cutoff = self.freq_cutoff

        if self.catalyst_enthalpies == [] or self.reactant_enthalpies == [] or self.product_enthalpies == [] or self.transition_state_enthalpies == []:
            self.get_thermodynamics(temp, pressure, freq_cutoff)

        if temp != self.temp or pressure != self.pressure or freq_cutoff != self.freq_cutoff:
            self.get_thermodynamics(temp, pressure, freq_cutoff)

        catalyst_enthalpies = self.catalyst_enthalpies
        reactant_enthalpies = self.reactant_enthalpies
        product_enthalpies = self.product_enthalpies
        transition_state_enthalpies = self.transition_state_enthalpies

        return catalyst_enthalpies, reactant_enthalpies, product_enthalpies, transition_state_enthalpies

    def get_entropies(self, temp=None, pressure=None, freq_cutoff=None):

        if temp is None:
            temp = self.temp
        if pressure is None:
            pressure = self.pressure
        if freq_cutoff is None:
            freq_cutoff = self.freq_cutoff

        if self.catalyst_entropies == [] or self.reactant_entropies == [] or self.product_entropies == [] or self.transition_state_entropies == []:
            self.get_thermodynamics(temp, pressure, freq_cutoff)

        if temp != self.temp or pressure != self.pressure or freq_cutoff != self.freq_cutoff:
            self.get_thermodynamics(temp, pressure, freq_cutoff)

        catalyst_entropies = self.catalyst_entropies
        reactant_entropies = self.reactant_entropies
        product_entropies = self.product_entropies
        transition_state_entropies = self.transition_state_entropies

        return catalyst_entropies, reactant_entropies, product_entropies, transition_state_entropies


if __name__ == '__main__':


    gauss = Gaussian(
            ['tests/A_0466_L-butyl.xyz', 'tests/A_0466_R-butyl.xyz'],
            ['tests/A_0466_L-butyl-R-ethylene.xyz', 'tests/A_0466_R-butyl-L-ethylene.xyz'],
            [],
            ['tests/A_0466_LR-transition-state.xyz', 'tests/A_0466_RL-transition-state.xyz'],
            'A_0466')
    gauss.setup()
    #gauss.run()
    #gibbs_energies = gauss.get_gibbs_energies()
    #print(gibbs_energies)


import numpy
import os


class Gaussian():

    def __init__(self, catalyst_file_paths, reactant_file_paths, product_file_paths, file_type,
            charges=[0, 0, 0, 0], mults=[4, 4, 4, 4],
            prefix='', nproc=1, method='wB97XD', basis='Gen',
            gen_basis='Cr 0\nDef2TZVP\n****\nSi O C H 0\nTZVP\n****',
            frozen_atoms=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            scan_params='B  1 19 2.1 B\nB  1 19 F\nB 19 20 2.2 B\nB 19 20 F',
            scan_reverse=True
            ):

        catalysts = [self.import_cluster(file_path, file_type) for file_path in catalyst_file_paths]
        reactants = [self.import_cluster(file_path, file_type) for file_path in reactant_file_paths]
        products = [self.import_cluster(file_path, file_type) for file_path in product_file_paths]

        self.set_parameters(nproc, method, basis, gen_basis, frozen_atoms, scan_params)

        self.catalyst_energies = []
        self.catalyst_clusters = []
        self.catalyst_free_energies = []
        for i, cluster in enumerate(catalysts):
            optimized_energy, optimized_cluster = self.optimize_geometry('{:s}_B_{:d}'.format(prefix, i), cluster, charges[0], mults[0])
            free_energy = self.read_thermochemistry('{:s}_B_{:d}.log'.format(prefix, i)) 
            self.catalyst_energies.append(optimized_energy)
            self.catalyst_clusters.append(optimized_cluster)
            self.catalyst_free_energies.append(free_energy)
        self.reactant_energies = []
        self.reactant_clusters = []
        self.reactant_free_energies = []
        for i, cluster in enumerate(reactants):
            optimized_energy, optimized_cluster = self.optimize_geometry('{:s}_R_{:d}'.format(prefix, i), cluster, charges[1], mults[1])
            free_energy = self.read_thermochemistry('{:s}_R_{:d}.log'.format(prefix, i))
            self.reactant_energies.append(optimized_energy)
            self.reactant_clusters.append(optimized_cluster)
            self.reactant_free_energies.append(free_energy)
        self.product_energies = []
        self.product_clusters = []
        self.product_free_energies = []
        for i, cluster in enumerate(products):
            optimized_energy, optimized_cluster = self.optimize_geometry('{:s}_P_{:d}'.format(prefix, i), cluster, charges[3], mults[3])
            free_energy = self.read_thermochemistry('{:s}_P_{:d}.log'.format(prefix, i))
            self.product_energies.append(optimized_energy)
            self.product_clusters.append(optimized_cluster)
            self.product_free_energies.append(free_energy)

        if scan_reverse:
            reference_clusters = self.product_clusters
        else:
            reference_clusters = self.reactant_clusters
        self.scan_energies = []
        self.scan_clusters = []
        for i, cluster in enumerate(reference_clusters):
            scan_energies, scan_clusters = self.scan_parameters('{:s}_S_{:d}'.format(prefix, i), cluster, charges[2], mults[2])
            self.scan_energies.append(scan_energies)
            self.scan_clusters.append(scan_clusters)
        self.transition_energies = []
        self.transition_clusters = []
        self.transition_free_energies = []
        for i, (scan_energies, scan_clusters) in enumerate(zip(self.scan_energies, self.scan_clusters)):
            transition_energy, transition_cluster = self.optimize_transition_state('{:s}_T_{:d}'.format(prefix, i), scan_clusters[numpy.argmax(scan_energies)], charges[2], mults[2])
            free_energy = self.read_thermochemistry('{:s}_T_{:d}.log'.format(prefix, i))
            self.transition_energies.append(transition_energy)
            self.transition_clusters.append(transition_cluster)
            self.transition_free_energies.append(free_energy)

        return

    def import_cluster(self, file_path, file_type):
        from ase.io import read
        cluster = read(file_path, 0, file_type)
        return cluster

    def set_parameters(self, nproc=None, method=None, basis=None, gen_basis=None, frozen_atoms=None, scan_params=None):
        if nproc is not None:
            self.nproc = nproc
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
        return

    def optimize_geometry(self, label, cluster, charge, mult):

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()

        header = '''%NProcShared={nproc:d}
#n {method:s}/{basis:s} NoSymm SCF=XQC Opt=(MaxCycles=200) Freq

 {label:s}

{charge:d} {mult:d}
'''.format(nproc=self.nproc, method=self.method, basis=self.basis, label=label, charge=charge, mult=mult)
        body = ''
        for j, (X, coord) in enumerate(zip(atoms, coords)):
            if j+1 in self.frozen_atoms:
                frozen = -1
            else:
                frozen = 0
            body += '{X:s} {frozen:d} {x:f} {y:f} {z:f}\n'.format(X=X, frozen=frozen, x=coord[0], y=coord[1], z=coord[2])
        footer = '\n'
        if self.basis in ['gen', 'Gen', 'GEN']:
            footer += self.gen_basis + '\n\n'

        f = open('{:s}.com'.format(label), 'wt')
        f.write(header + body + footer)
        f.close()

        if os.path.exists('{:s}.log'.format(label)):
            if self.check_normal_termination('{:s}.log'.format(label)):
                print('already done')
            else:
                os.system('g16 {label:s}.com > {label:s}.log'.format(label=label))
        else:
            os.system('g16 {label:s}.com > {label:s}.log'.format(label=label))

        if self.check_normal_termination('{:s}.log'.format(label)):
            optimized_energies, optimized_clusters = self.read_optimized_geometries('{:s}.log'.format(label))
            return optimized_energies[-1], optimized_clusters[-1]
        else:
            return

    def scan_parameters(self, label, cluster, charge, mult):

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()

        header = '''%NProcShared={nproc:d}
#n PBEPBE/STO-3G NoSymm SCF=XQC Opt=(ModRedundant,Loose,MaxCycles=200)

 {label:s}

{charge:d} {mult:d}
'''.format(nproc=self.nproc, label=label, charge=charge, mult=mult)
        body = ''
        for j, (X, coord) in enumerate(zip(atoms, coords)):
            if j in self.frozen_atoms:
                frozen = -1
            else:
                frozen = 0
            body += '{X:s} {frozen:d} {x:f} {y:f} {z:f}\n'.format(X=X, frozen=frozen, x=coord[0], y=coord[1], z=coord[2])
        footer = '\n{:s}\n\n'.format(self.scan_params)
        if self.basis in ['gen', 'Gen', 'GEN']:
            footer += self.gen_basis + '\n\n'

        f = open('{:s}.com'.format(label), 'wt')
        f.write(header + body + footer)
        f.close()

        if os.path.exists('{:s}.log'.format(label)):
            print('already done')
        else:
            os.system('g16 {label:s}.com > {label:s}.log'.format(label=label))

        scan_energies, scan_clusters = self.read_optimized_geometries('{:s}.log'.format(label))

        return scan_energies, scan_clusters

    def optimize_transition_state(self, label, cluster, charge, mult):

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()

        header = '''%NProcShared={nproc:d}
#n {method:s}/{basis:s} NoSymm SCF=XQC Opt=(TS,NoEigen,CalcFC,MaxCycles=200) Freq

 {label:s}

{charge:d} {mult:d}
'''.format(nproc=self.nproc, method=self.method, basis=self.basis, label=label, charge=charge, mult=mult)
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

        f = open('{:s}.com'.format(label), 'wt')
        f.write(header + body + footer)
        f.close()

        if os.path.exists('{:s}.log'.format(label)):
            if self.check_normal_termination('{:s}.log'.format(label)):
                print('already done')
            else:
                os.system('g16 {label:s}.com > {label:s}.log'.format(label=label))
        else:
            os.system('g16 {label:s}.com > {label:s}.log'.format(label=label))

        if self.check_normal_termination('{:s}.log'.format(label)):
            transition_energies, transition_clusters = self.read_optimized_geometries('{:s}.log'.format(label))
            return transition_energies[-1], transition_clusters[-1]
        else:
            return

    def check_normal_termination(self, file_path):
        f = open(file_path, 'rt')
        for line in f:
            continue
        last_line = line
        f.close()
        if last_line.strip().startswith('Normal termination'):
            return True
        else:
            return False

    def read_optimized_geometries(self, file_path):
        from ase import Atoms
        f = open(file_path, 'rt')
        energies = []
        clusters = []
        status = 0
        for line in f:
            if status == 0:
                if line.strip().startswith('SCF Done:'):
                    energy = float(line.split()[4])
                elif line.strip().startswith('!   Optimized Parameters   !') or line.strip().startswith('! Non-Optimized Parameters !'):
                    atoms = []
                    coords = []
                    status = 1
            elif status == 1:
                if line.strip().startswith('Input orientation:'):
                    status = 2
            elif status == 2:
                if line.strip().startswith('-----'):
                    status = 3
            elif status == 3:
                if line.strip().startswith('-----'):
                    status = 4
            elif status == 4:
                if line.strip().startswith('-----'):
                    energies.append(energy)
                    clusters.append(Atoms(atoms, coords))
                    status = 0
                else:
                    atoms.append(int(line.split()[1]))
                    coords.append([float(x) for x in line.split()[-3:]])
        f.close()
        return energies, clusters

    def read_thermochemistry(self, file_path, new_algorithm=True, new_constants=True):

        if new_algorithm:

            Ha = 627.5094740631

            T = numpy.nan
            H_tot = numpy.nan
            S_tot = numpy.nan
            H_trans = numpy.nan
            S_trans = numpy.nan
            H_rot = numpy.nan
            S_rot = numpy.nan

            f = open(file_path, 'rt')
            status = 0
            for line in f:
                if status == 0:
                    if line.strip().startswith('Temperature'):
                        T = float(line.split()[1])
                        status = 1
                elif status == 1:
                    if line.strip().startswith('Sum of electronic and thermal Enthalpies='):
                        H_tot = float(line.split()[-1])
                        status = 2
                elif status == 2:
                    if line.strip().startswith('E (Thermal)'):
                        status = 3
                elif status == 3:
                    if line.strip().startswith('Total'):
                        S_tot = float(line.split()[-1])
                    elif line.strip().startswith('Translational'):
                        H_trans = float(line.split()[1])
                        S_trans = float(line.split()[-1])
                    elif line.strip().startswith('Rotational'):
                        H_rot = float(line.split()[1])
                        S_rot = float(line.split()[-1])
                    elif line.strip().startswith('Q'):
                        status = 4
                        break
            f.close()

            H = H_tot - (H_trans + H_rot) / Ha
            S = (S_tot - S_trans - S_rot) / (1000 * Ha)
            G = H - T * S

            return G

        else:

            if new_constants:
                kB = 1.380649e-23 / 4.3597447222071e-18
                K = 6.62607015e-34*2.99792458e+8/1.380649e-23 * 100.0
            else:
                kB = 8.314 / 2625.499 / 1000
                K = 1.4387772494045046

            T = numpy.nan
            E_e = numpy.nan
            log_Q_e = numpy.nan
            freqs = []

            f = open(file_path, 'rt')
            status = 0
            for line in f:
                if status == 0:
                    if line.strip().startswith('SCF Done:'):
                        E_e = float(line.split()[4])
                    elif line.strip().startswith('Harmonic frequencies'):
                        status = 1
                elif status == 1:
                    if line.strip().startswith('Frequencies'):
                        freqs += [float(x) for x in line.split()[-3:]]
                    elif line.strip().startswith('- Thermochemistry -'):
                        status = 2
                elif status == 2:
                    if line.strip().startswith('Temperature'):
                        T = float(line.split()[1])
                    elif line.strip().endswith('Ln(Q)'):
                        status = 3
                elif status == 3:
                    if line.strip().startswith('Electronic'):
                        log_Q_e = float(line.split()[-1])
                    elif line.strip().startswith('-----'):
                        status = 4
                        break
            f.close()
            freqs = numpy.array(freqs)

            T_v = freqs[freqs > 0.0] * K
            E_v = kB * numpy.sum( T_v * ( 1/2 + 1/(numpy.exp(T_v/T) - 1) ) )
            S_v = kB * numpy.sum( (T_v/T) / (numpy.exp(T_v/T) - 1) - numpy.log(1 - numpy.exp(-T_v/T)) )

            S_e = kB * log_Q_e

            H = E_e + E_v + kB * T
            S = S_e + S_v
            G = H - T * S

            return G

    def get_free_energies(self):
        return self.catalyst_free_energies, self.reactant_free_energies, self.transition_free_energies, self.product_free_energies

if __name__ == '__main__':

    gauss = Gaussian(['ammonia.xyz'], ['ammonia_borane.xyz'], ['borazane.xyz'], 'xyz',
            charges=[0, 0, 0, 0], mults=[1, 1, 1, 1],
            prefix='NH3BH3', nproc=4, method='PBEPBE', basis='3-21G', frozen_atoms=[2, 3, 4], scan_params='B 1 5 S 5 0.2')
    print(gauss.get_free_energies())
    print('new algorithm', gauss.read_thermochemistry('1_b.log', new_algorithm=True))
    print('new constants', gauss.read_thermochemistry('1_b.log', new_algorithm=False, new_constants=True))
    print('old constants', gauss.read_thermochemistry('1_b.log', new_algorithm=False, new_constants=False))
    print('salman script', -594.3547231345688)


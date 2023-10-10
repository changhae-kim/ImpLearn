import numpy
import os

from ase import Atoms
from ase.io import read, write

#from .graft import MatchCoords


def check_normal_termination(file_path):
    status = False
    f = open(file_path,'rt')
    for line in f:
        if line.strip().startswith('normal termination of xtb'):
            status = True
            break
    f.close()
    return status

def get_cluster(file_path):
    energy = None
    clsuter = None
    f = open(file_path, 'rt')
    status = 0
    n_atoms = 0
    for line in f:
        if status == 0:
            n_atoms = int(line)
            atoms = []
            coords = []
            status = 1
        elif status == 1:
            energy = float(line.split()[1])
            status = 2
        elif status == 2:
            atoms.append(line.split()[0])
            coords.append([float(x) for x in line.split()[1:]])
            if len(atoms) >= n_atoms:
                cluster = Atoms(atoms, coords)
                status = 0
    f.close()
    return energy, cluster


class xTB():

    def __init__(self, file_paths, prefix,
            charges=0, spins=3,
            n_proc=24, method='GFN2-xTB',
            constraints='$fix\n  atoms: 1-4\n$constrain\n  force constant=0.5\n  distance: 10, 11, auto\n$end\n',
            opt_thresh='vtight',
            e_window=0.00956161, r_thresh=0.125,
            exclude_atoms=[0, 1, 2, 3],
            exclude_elements='H',
            degeneracies=1
            ):

        self.file_paths = file_paths
        self.prefix = prefix

        n_struct = len(self.file_paths)

        if isinstance(charges, int):
            self.charges = [charges] * n_struct
        else:
            self.charges = charges

        if isinstance(spins, int):
            self.spins = [spins] * n_struct
        else:
            self.spins = spins

        self.n_proc = n_proc
        self.method = method

        if isinstance(constraints, str):
            self.constraints = [constraints] * n_struct
        else:
            self.constraints = constraints

        self.opt_thresh = opt_thresh

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
        elif exclude_atoms == [] or isinstance(exclude_elements[0], str):
            self.exclude_elements = [exclude_elements] * n_struct
        else:
            self.exclude_elements = exclude_elements

        if isinstance(degeneracies, int):
            self.degeneracies = [degeneracies for i in range(n_struct)]
        else:
            self.degeneracies = degeneracies

        self.labels = [[] for i in range(n_struct)]
        self.workspaces = [[] for i in range(n_struct)]
        self.clusters = [[] for i in range(n_struct)]
        self.energies = [[] for i in range(n_struct)]

        return

    def setup(self):

        script = '''#!/bin/bash
#$ -cwd
#$ -V
#$ -N xtb-{label:s}
#$ -pe default {n_proc:d}

export OMP_NUM_THREADS={n_proc:d},1
/home/cakim2/conda/envs/petersgroup/bin/xtb {method:s}inp.xyz --chrg {charge:d} --uhf {spin:d} {opt:s}{inp:s}>xtb.log 2>&1

'''

        if self.method == 'GFN0-xTB':
            method = '--gfn 0 '
        elif self.method == 'GFN1-xTB':
            method = '--gfn 1 '
        elif self.method == 'GFN-FF':
            method = '--gfnff '
        else:
            method = ''

        if self.opt_thresh == 'noopt':
            opt = ''
        elif self.opt_thresh == 'tight':
            opt = '--opt tight '
        elif self.opt_thresh == 'vtight':
            opt = '--opt vtight '
        else:
            opt = '--opt '

        n_struct = len(self.file_paths)
        for i in range(n_struct):
            clusters = read(self.file_paths[i], ':', 'xyz')
            n_digits = str(len(str(len(clusters))))
            file_name = os.path.basename(self.file_paths[i])
            for j, _ in enumerate(clusters):
                label = ('{:s}.{:0' + n_digits + 'd}').format(file_name.rsplit('.', 1)[0], j)
                workspace = os.path.join(self.prefix, label)
                if not os.path.exists(workspace):

                    cwd = os.getcwd()

                    os.mkdir(workspace)
                    os.chdir(workspace)

                    write('inp.xyz', clusters[j], 'xyz')

                    if self.constraints[i] != '':
                        f = open('xtb.inp', 'wt')
                        f.write(self.constraints[i])
                        f.close()
                        inp = '--input xtb.inp '
                    else:
                        inp = ''

                    f = open('xtb.sh', 'wt')
                    f.write(script.format(label=label, n_proc=self.n_proc, method=method, charge=self.charges[i], spin=self.spins[i], opt=opt, inp=inp))
                    f.close()

                    os.chdir(cwd)

                if label not in self.labels[i]:
                    self.labels[i].append(label)
                if workspace not in self.workspaces[i]:
                    self.workspaces[i].append(workspace)

        return

    def run(self, dry_run=False):

        n_struct = len(self.file_paths)
        for i in range(n_struct):
            for workspace in self.workspaces[i]:

                output = os.path.join(workspace, 'xtb.log')

                if not os.path.exists(output):
                    if not dry_run:
                        cwd = os.getcwd()
                        os.chdir(workspace)
                        os.system('/bin/bash xtb.sh')
                        os.chdir(cwd)

                if os.path.exists(output):
                    status = check_normal_termination(output)
                    if status == True:
                        if self.opt_thresh == 'noopt':
                            energy = 0.0
                            f = open(os.path.join(workspace, 'xtb.log'), 'rt')
                            for line in f:
                                if line.strip().startswith('| TOTAL ENERGY'):
                                    energy = float(line.split()[-3])
                                    break
                            f.close()
                            cluster = read(os.path.join(workspace, 'inp.xyz'), 0, 'xyz')
                        else:
                            energy, cluster = get_cluster(os.path.join(workspace, 'xtbopt.xyz'))
                        self.energies[i].append(energy)
                        self.clusters[i].append(cluster)
                    else:
                        print(workspace, 'Error termination')
                else:
                    print(workspace, 'No output')

        return

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

        n_struct = len(self.file_paths)
        for i in range(n_struct):
            sorted_degeneracies = []
            sorted_labels = []
            sorted_workspaces = []
            sorted_energies = []
            sorted_clusters = []
            if reorder:
                iterator = numpy.argsort(self.energies[i])
            else:
                iterator = [m for m, _ in enumerate(self.energies[i], kind='stable')]
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
                sorted_labels.append(self.labels[i][m])
                sorted_workspaces.append(self.workspaces[i][m])
                sorted_energies.append(self.energies[i][m])
                sorted_clusters.append(self.clusters[i][m])
            self.degeneracies[i] = sorted_degeneracies
            self.labels[i] = sorted_labels
            self.workspaces[i] = sorted_workspaces
            self.energies[i] = sorted_energies
            self.clusters[i] = sorted_clusters

        return


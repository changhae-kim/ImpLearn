import os

from ase.io import read, write


class xTB():

    def __init__(self, file_paths, prefix,
            charges=0, mults=3,
            n_proc=24,
            constraints='$fix\n  atoms: 1-4\n$constrain\n  force constant=0.5\n  distance: 10, 11, auto\n$end\n'
            ):

        self.file_paths = file_paths
        self.prefix = prefix

        n_struct = len(self.file_paths)

        if isinstance(charges, int):
            self.charges = [charges] * n_struct
        else:
            self.charges = charges

        if isinstance(mults, int):
            self.mults = [mults] * n_struct
        else:
            self.mults = mults

        self.n_proc = n_proc

        if isinstance(constraints, str):
            self.constraints = [constraints] * n_struct
        else:
            self.constraints = constraints

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
/home/cakim2/conda/envs/petersgroup/bin/xtb inp.xyz --chrg {charge:d} --uhf {mult:d} --opt vtight --input xtb.inp >xtb.log 2>&1

'''

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

                    f = open('xtb.inp', 'wt')
                    f.write(self.constraints[i])
                    f.close()

                    f = open('xtb.sh', 'wt')
                    f.write(script.format(label=label, n_proc=self.n_proc, charge=self.charges[i], mult=self.mults[i]))
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
                    energy, cluster = get_cluster(os.path.join(workspace, 'xtbopt.xyz'))
                    self.energies[i].append(energy)
                    self.clusters[i].append(cluster)
                else:
                    print(workspace, 'No output')

        return

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


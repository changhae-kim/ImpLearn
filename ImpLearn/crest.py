import os

from ase import Atoms


def check_normal_termination(file_path):
    last_line = ''
    f = open(file_path,'rt')
    for line in f:
        last_line = line
    f.close()
    if last_line.strip().startswith('CREST terminated normally.'):
        return True
    else:
        return False

def get_conformers(file_path):
    energies = []
    clusters = []
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
            energies.append(float(line.split()[0]))
            status = 2
        elif status == 2:
            atoms.append(line.split()[0])
            coords.append([float(x) for x in line.split()[1:]])
            if len(atoms) >= n_atoms:
                clusters.append(Atoms(atoms, coords))
                status = 0
    f.close()
    return energies, clusters

def get_degeneracies(file_path):
    degeneracies = []
    f = open(file_path, 'rt')
    status = 0
    n_conf = 0
    for line in f:
        if status == 0:
            if line.strip().startswith('total number unique points'):
                n_conf = int(line.split()[-1])
                status = 1
        elif status == 1:
            status = 2
        elif status == 2:
            row = line.split()
            if len(row) > 6:
                degeneracies.append(int(row[6]))
            if int(row[0]) == n_conf:
                status = 0
    f.close()
    return degeneracies


class Crest():

    def __init__(self, file_paths, prefix,
            charges=0, spins=3,
            n_proc=24,
            constraints='$constrain\n  atoms: 1-4\n  force constant=0.5\n  reference=coord.ref\n$metadyn\n  atoms: 5-11\n$end\n'
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

        if isinstance(constraints, str):
            self.constraints = [constraints] * n_struct
        else:
            self.constraints = constraints

        self.labels = []
        self.workspaces = []

        self.degeneracies = [[] for i in range(n_struct)]
        self.conformers = [[] for i in range(n_struct)]
        self.conformer_energies = [[] for i in range(n_struct)]
        self.rotamers = [[] for i in range(n_struct)]
        self.rotamer_energies = [[] for i in range(n_struct)]

        return

    def setup(self):

        script = '''#!/bin/bash
#$ -cwd
#$ -V
#$ -N crest-{label:s}
#$ -pe default {n_proc:d}

/home/cakim2/conda/envs/petersgroup/bin/crest coord --T {n_proc:d} --chrg {charge:d} --uhf {spin:d} --cinp constraints.inp --subrmsd --noreftopo >crest.log 2>&1

'''

        n_struct = len(self.file_paths)
        for i in range(n_struct):
            file_name = os.path.basename(self.file_paths[i])
            label = file_name.rsplit('.', 1)[0]
            workspace = os.path.join(self.prefix, label)
            if not os.path.exists(workspace):

                cwd = os.getcwd()

                os.mkdir(workspace)
                os.system('cp {:s} {:s}'.format(self.file_paths[i], workspace))
                os.chdir(workspace)

                os.system('crest {:s} --constrain 1-4 >/dev/null 2>&1'.format(file_name))
                os.remove('.xcontrol.sample')

                f = open('constraints.inp', 'wt')
                f.write(self.constraints[i])
                f.close()

                f = open('crest.sh', 'wt')
                f.write(script.format(label=label, n_proc=self.n_proc, charge=self.charges[i], spin=self.spins[i]))
                f.close()

                os.chdir(cwd)

            if label not in self.labels:
                self.labels.append(label)
            if workspace not in self.workspaces:
                self.workspaces.append(workspace)

        return

    def run(self, dry_run=False):

        n_struct = len(self.file_paths)
        for i in range(n_struct):

            output = os.path.join(self.workspaces[i], 'crest.log')

            if not os.path.exists(output):
                if not dry_run:
                    cwd = os.getcwd()
                    os.chdir(workspace)
                    os.system('/bin/bash crest.sh')
                    os.chdir(cwd)

            if os.path.exists(output):
                status = check_normal_termination(output)
                if status == True:
                    self.degeneracies[i] = get_degeneracies(output)
                    self.conformer_energies[i], self.conformers[i] = get_conformers(os.path.join(self.workspaces[i], 'crest_conformers.xyz'))
                    self.rotamer_energies[i], self.rotamers[i] = get_conformers(os.path.join(self.workspaces[i], 'crest_rotamers.xyz'))
                else:
                    print(self.workspaces[i], 'Error termination')
            else:
                print(self.workspaces[i], 'No output')

        return


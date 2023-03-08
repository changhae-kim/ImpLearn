import numpy
import os


def check_normal_termination(self, file_path):
    last_line = ''
    f = open(file_path,'rt')
    for line in f:
        last_line = line
    f.close()
    if last_line.strip().startswith('CREST terminated normally.'):
        return True
    else:
        return False


class Crest():

    def __init__(self, file_paths, prefix,
            charges=0, mults=3,
            n_proc=24,
            constraints='$constrain\n  atoms: 1-4\n  force constant=10.0\n  reference=coord.ref\n$metadyn\n  atoms: 5-11\n$end\n'
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

        return

    def setup(self):

        script = '''#!/bin/bash
#$ -cwd
#$ -V
#$ -N crest-{label:s}
#$ -pe default {n_proc:d}

/home/cakim2/conda/envs/petersgroup/bin/crest coord --T {n_proc:d} --chrg {charge:d} --uhf {mult:d} --cinp constraints.inp --subrmsd --noreftopo >crest.log 2>&1

'''

        n_struct = len(self.file_paths)
        for i in range(n_struct):
            file_name = os.path.basename(self.file_paths[i])
            label = file_name.rsplit('.', 1)[0]
            work_dir = os.path.join(self.prefix, label)
            if not os.path.exists(work_dir):

                cwd = os.getcwd()

                os.mkdir(work_dir)
                os.system('cp {:s} {:s}'.format(self.file_paths[i], work_dir))

                os.chdir(work_dir)

                os.system('crest {:s} --constrain 1-4 >/dev/null 2>&1'.format(file_name))
                os.remove('.xcontrol.sample')

                f = open('constraints.inp', 'wt')
                f.write(self.constraints[i])
                f.close()

                f = open('crest.sh', 'wt')
                f.write(script.format(label=label, n_proc=self.n_proc, charge=self.charges[i], mult=self.mults[i]))
                f.close()

                os.chdir(cwd)

        return

    def run(self, dry_run=False):

        n_struct = len(self.file_paths)
        for i in range(n_struct):
            label = os.path.basename(self.file_paths[i]).rsplit('.', 1)[0]
            work_dir = os.path.join(self.prefix, label)
            output = os.path.join(work_dir, 'crest.log')

            if not os.path.exists(output):
                if not dry_run:
                    cwd = os.getcwd()
                    os.chdir(work_dir)
                    os.system('/bin/bash crest.sh')
                    os.chdir(cwd)

            if os.path.exists(output):
                status == check_normal_termination(output)
                if status == False:
                    print(work_dir, 'Error termination')
            else:
                print(work_dir, 'No output')

        return


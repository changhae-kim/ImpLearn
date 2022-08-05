import numpy

from ase import Atoms


def check_normal_termination(file_path):
    last_line = ''
    f = open(file_path, 'rt')
    for line in f:
        last_line = line
    f.close()
    if last_line.strip().startswith('Normal termination'):
        return True
    else:
        return False

def read_geometry_optimization(file_path):
    f = open(file_path, 'rt')
    n_cycles = 0
    energies = []
    clusters = []
    status = 0
    for line in f:
        if status == 0:
            if line.strip().startswith('Input orientation:'):
                atoms = []
                coords = []
                status = 1
            elif line.strip().startswith('SCF Done:'):
                n_cycles += 1
                energy = float(line.split()[4])
            elif line.strip().startswith('!   Optimized Parameters   !') or line.strip().startswith('! Non-Optimized Parameters !'):
                energies.append(energy)
                clusters.append(Atoms(atoms, coords))
        elif status == 1:
            if line.strip().startswith('-----'):
                status = 2
        elif status == 2:
            if line.strip().startswith('-----'):
                status = 3
        elif status == 3:
            if line.strip().startswith('-----'):
                status = 0
            else:
                atoms.append(int(line.split()[1]))
                coords.append([float(x) for x in line.split()[-3:]])
    f.close()
    if energies == [] and n_cycles > 1:
        energies.append(energy)
        clusters.append(Atoms(atoms, coords))
    return energies, clusters

def read_thermochemistry(file_path):

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
    S = (S_tot - S_trans - S_rot) / Ha / 1000.0
    G = H - T * S

    return G

def read_thermochemistry_salman(file_path, new_constants=False):

    if new_constants:
        kB = 1.380649e-23 / 4.3597447222071e-18
        K = 6.62607015e-34 * 2.99792458e+8 / 1.380649e-23 * 100.0
    else:
        kB = 8.314 / 2625.499 / 1000.0
        K = 1.4387772494045046

    T = numpy.nan
    E_e = numpy.nan
    G_total = numpy.nan
    log_Q_e = numpy.nan
    log_Q_t = numpy.nan
    log_Q_r = numpy.nan
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
            elif line.strip().startswith('Sum of electronic and thermal Free Energies='):
                G_total = float(line.split()[-1])
            elif line.strip().endswith('Ln(Q)'):
                status = 3
        elif status == 3:
            if line.strip().startswith('Electronic'):
                log_Q_e = float(line.split()[-1])
            elif line.strip().startswith('Translational'):
                log_Q_t = float(line.split()[-1])
            elif line.strip().startswith('Rotational'):
                log_Q_r = float(line.split()[-1])
            elif line.strip().startswith('-----'):
                status = 4
                break
    f.close()
    freqs = numpy.array(freqs)

    T_v = freqs[freqs > 0.0] * K
    E_v = kB * numpy.sum( T_v * ( 0.5 + 1.0/(numpy.exp(T_v/T) - 1.0) ) )
    S_v = kB * numpy.sum( (T_v/T) / (numpy.exp(T_v/T) - 1.0) - numpy.log(1.0 - numpy.exp(-T_v/T)) )

    S_e = kB * log_Q_e

    H = E_e + E_v + kB * T
    S = S_e + S_v
    G = G_total + kB * T * (1.0 + log_Q_t + log_Q_r)

    return G

def check_geometry(cluster, criteria):
    status = True
    for (i, j), (dmin, dmax) in criteria.items():
        distance = cluster.get_distance(i, j)
        if distance < dmin or distance > dmax:
            status = False
            break
    return status


if __name__ == '__main__':

    '''
    print('new algorithm', read_thermochemistry('tests/1_b.log'))
    print('old algorithm', read_thermochemistry_salman('tests/1_b.log', new_constants=True))
    print('old constants', read_thermochemistry_salman('tests/1_b.log', new_constants=False))
    print('salman script', -594.3547231345688)
    '''



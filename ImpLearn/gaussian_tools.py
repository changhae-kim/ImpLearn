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
    elif last_line.strip().startswith('File lengths'):
        return False
    else:
        return 2

def check_geometry(cluster, criteria):
    status = True
    for (i, j), (dmin, dmax) in criteria.items():
        distance = cluster.get_distance(i, j)
        if distance < dmin or distance > dmax:
            status = False
            break
    return status

def read_geom_opt(file_path):
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
            elif line.strip().startswith('!   Optimized Parameters   !') or (line.strip().startswith('Normal termination') and energies == []):
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

def read_irc(file_path):
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
            elif line.strip().startswith('# OF POINTS ALONG THE PATH') or (line.strip().startswith('Normal termination') and energies == []):
                energies.append(energy)
                clusters.append(Atoms(atoms, coords))
            elif line.strip().startswith('Calculation of FORWARD path complete.') or line.strip().startswith('Calculation of REVERSE path complete.'):
                energies = list(reversed(energies))
                clusters = list(reversed(clusters))
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

def read_thermochem(file_path, temp=None, pressure=None,
        elec=True, trans=False, rot=False, vib=True,
        freq_cutoff=0.0, verbose=False):

    kB = 1.380649e-23 / 4.3597447222071e-18
    K = 6.62607015e-34 * 2.99792458e+8 / 1.380649e-23 * 100.0

    T_0 = numpy.nan
    P_0 = numpy.nan
    E_e = numpy.nan
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
                freqs += [float(x) for x in line.split()[2:]]
            elif line.strip().startswith('- Thermochemistry -'):
                status = 2
        elif status == 2:
            if line.strip().startswith('Temperature'):
                T_0 = float(line.split()[1])
                P_0 = float(line.split()[4])
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

    if temp is None:
        T = T_0
    else:
        T = temp
    if pressure is None:
        P = P_0
    else:
        P = pressure

    S_e = kB * log_Q_e

    E_t = 1.5 * kB * T
    S_t = kB * (log_Q_t + 2.5 * numpy.log(T/T_0) - numpy.log(P/P_0) + 1.0 + 1.5)

    E_r = 1.5 * kB * T
    S_r = kB * (log_Q_r + 1.5 * numpy.log(T/T_0) + 1.5)

    freqs = numpy.array(freqs)
    T_v = freqs[freqs > freq_cutoff] * K
    E_v = kB * numpy.sum( T_v * ( 0.5 + 1.0/(numpy.exp(T_v/T) - 1.0) ) )
    S_v = kB * numpy.sum( (T_v/T) / (numpy.exp(T_v/T) - 1.0) - numpy.log(1.0 - numpy.exp(-T_v/T)) )

    E = elec * E_e + trans * E_t + rot * E_r + vib * E_v
    S = elec * S_e + trans * S_t + rot * S_r + vib * S_v
    H = E + kB * T
    G = H - T * S

    if verbose:
        return E_e, H, S, G
    else:
        return G

def read_thermochem_salman(file_path, new_constants=False):

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


if __name__ == '__main__':


    print('salman script', -594.3547231345688)
    print('salman algorithm with old constants', read_thermochem_salman('tests/1_b.log', new_constants=False))
    print('salman algorithm with new constants', read_thermochem_salman('tests/1_b.log', new_constants=True))
    print('new algorithm', read_thermochem('tests/1_b.log'))

    elec, trans, rot, vib = True, True, True, True
    print('act 300 K 1 atm', read_thermochem('tests/1_b.log', elec=elec, trans=trans, rot=rot, vib=vib))
    print('est 600 K 1 atm', read_thermochem('tests/1_b.log', temp=600.0, elec=elec, trans=trans, rot=rot, vib=vib))
    print('act 600 K 1 atm', read_thermochem('tests/1_b_T.log', elec=elec, trans=trans, rot=rot, vib=vib))
    print('est 300 K 2 atm', read_thermochem('tests/1_b.log', pressure=2.0, elec=elec, trans=trans, rot=rot, vib=vib))
    print('act 300 K 2 atm', read_thermochem('tests/1_b_P.log', elec=elec, trans=trans, rot=rot, vib=vib))
    print('est 600 K 2 atm', read_thermochem('tests/1_b.log', temp=600.0, pressure=2.0, elec=elec, trans=trans, rot=rot, vib=vib))
    print('act 600 K 2 atm', read_thermochem('tests/1_b_TP.log', elec=elec, trans=trans, rot=rot, vib=vib))



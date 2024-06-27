import numpy

from ase import Atoms


def write_clusters(file_path, clusters, comments=None):

    if comments is None:
        comments = ['' for cluster in clusters]

    lines = []
    for i, _ in enumerate(clusters):
        atoms = clusters[i].get_chemical_symbols()
        coords = clusters[i].get_positions()
        lines.append('{:d}\n'.format(len(atoms)))
        lines.append('{:s}\n'.format(comments[i]))
        for j, _ in enumerate(atoms):
            lines.append('{:2s} {:9.6f} {:9.6f} {:9.6f}\n'.format(atoms[j], *coords[j]))

    f = open(file_path, 'wt')
    f.writelines(lines)
    f.close()

    return

def check_imaginary_frequency(file_path):
    imgfrq = 0
    f = open(file_path, 'rt')
    for line in f:
        if line.rstrip().endswith('ignored.'):
            imgfrq = int(line.split()[0])
            break
    f.close()
    return imgfrq

def check_normal_termination(file_path):
    f = open(file_path, 'rt')
    lines = f.readlines()
    f.close()
    if lines[-1].strip().startswith('Normal termination'):
        return 0
    elif lines[-1].strip().startswith('File lengths'):
        return 1
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

    energies = []
    clusters = []

    f = open(file_path, 'rt')
    status = 0
    for line in f:
        if status == 0:
            if line.strip().startswith('Input orientation:'):
                atoms = []
                coords = []
                status = 1
            elif line.strip().startswith('SCF Done:'):
                energy = float(line.split()[4])
            elif line.strip().startswith('Counterpoise corrected energy ='):
                energy = float(line.split()[-1])
            elif line.strip().startswith('!   Optimized Parameters   !'):
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

    if energies == []:
        energies.append(energy)
        clusters.append(Atoms(atoms, coords))

    return energies, clusters

def read_irc(file_path):

    energies = []
    clusters = []

    f = open(file_path, 'rt')
    status = 0
    for line in f:
        if status == 0:
            if line.strip().startswith('Input orientation:'):
                atoms = []
                coords = []
                status = 1
            elif line.strip().startswith('SCF Done:'):
                energy = float(line.split()[4])
            elif line.strip().startswith('Counterpoise corrected energy ='):
                energy = float(line.split()[-1])
            elif line.strip().startswith('# OF POINTS ALONG THE PATH'):
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

    if energies == []:
        energies.append(energy)
        clusters.append(Atoms(atoms, coords))

    return energies, clusters

def read_orbitals(file_path):

    sf = 6
    eigvals = []

    f = open(file_path, 'rt')
    status = 0
    for line in f:
        if status == 0:
            if line.strip().startswith('Population analysis'):
                alpha_occ = []
                alpha_vir = []
                beta_occ = []
                beta_vir = []
                status = 1
        if status == 1:
            if line.strip().startswith('Condensed to atoms'):
                eigvals.append([alpha_occ, alpha_vir, beta_occ, beta_vir])
                status = 0
            elif line.strip().startswith('Alpha  occ.'):
                data = line[28:]
                ii = 0
                for i, x in enumerate(data):
                    if x == '.':
                        alpha_occ.append(float(data[ii:i+sf]))
                        ii = i+sf
            elif line.strip().startswith('Alpha virt.'):
                data = line[28:]
                ii = 0
                for i, x in enumerate(data):
                    if x == '.':
                        alpha_vir.append(float(data[ii:i+sf]))
                        ii = i+sf
            elif line.strip().startswith('Beta  occ.'):
                data = line[28:]
                ii = 0
                for i, x in enumerate(data):
                    if x == '.':
                        beta_occ.append(float(data[ii:i+sf]))
                        ii = i+sf
            elif line.strip().startswith('Beta virt.'):
                data = line[28:]
                ii = 0
                for i, x in enumerate(data):
                    if x == '.':
                        beta_vir.append(float(data[ii:i+sf]))
                        ii = i+sf
    f.close()

    return eigvals

def read_vib_modes(file_path):

    n_atoms = 0
    eigvals = []
    eigvecs = []

    f = open(file_path, 'rt')
    status = 0
    for line in f:
        if status == 0:
            if line.strip().startswith('Input orientation:'):
                status = 1
        elif status == 1:
            if line.strip().startswith('-----'):
                status = 2
        elif status == 2:
            if line.strip().startswith('-----'):
                status = 3
        elif status == 3:
            if line.strip().startswith('-----'):
                status = 10
            else:
                n_atoms += 1
        elif status == 10:
            if line.strip().startswith('Harmonic frequencies'):
                status = 11
        elif status == 11:
            if line.strip().startswith('-----'):
                status = 20
            elif line.strip().startswith('Frequencies'):
                evals = [float(x) for x in line.split()[2:]]
                status = 12
        elif status == 12:
            if line.strip().startswith('Atom'):
                evecs = [numpy.zeros((n_atoms, 3)) for x in evals]
                status = 13
        elif status == 13:
            data = line.split()
            if len(data) < 4:
                eigvals += evals
                eigvecs += evecs
                status = 11
            else:
                n = int(data[0])-1
                for i, _ in enumerate(evals):
                    for j in range(3):
                        evecs[i][n,j] = float(data[2+3*i+j])
    f.close()

    return eigvals, eigvecs

def read_thermochem(file_path, temp=None, pressure=None, vib_cutoff=0.0,
        elec=True, trans=False, rot=False, vib=True,
        gibbs_only=True):

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
            elif line.strip().startswith('Counterpoise corrected energy ='):
                energy = float(line.split()[-1])
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
    T_v = freqs[freqs > vib_cutoff] * K
    E_v = kB * numpy.sum( T_v * ( 0.5 + 1.0/(numpy.exp(T_v/T) - 1.0) ) )
    S_v = kB * numpy.sum( (T_v/T) / (numpy.exp(T_v/T) - 1.0) - numpy.log(1.0 - numpy.exp(-T_v/T)) )

    E = elec * E_e + trans * E_t + rot * E_r + vib * E_v
    S = elec * S_e + trans * S_t + rot * S_r + vib * S_v
    H = E + kB * T
    G = H - T * S

    if gibbs_only:
        return G
    else:
        return E_e, H, S, G

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
            elif line.strip().startswith('Counterpoise corrected energy ='):
                energy = float(line.split()[-1])
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


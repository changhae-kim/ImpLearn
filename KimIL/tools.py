import numpy

from ase import Atoms


def calculate_k(G_r, G_ts, T):
    kB = 1.380649e-23 / 4.3597447222071e-18
    h = 6.62607015e-34 / 4.3597447222071e-18
    G_r = numpy.array(G_r)
    G_ts = numpy.array(G_ts)
    k = (kB * T / h) * numpy.exp(-(G_ts - G_r) / (kB * T))
    return k

def calculate_K(G_a, G_b, G_r, T):
    kB = 1.380649e-23 / 4.3597447222071e-18
    G_a = numpy.array(G_a)
    G_b = numpy.array(G_b)
    G_r = numpy.array(G_r)
    K = numpy.exp(-(G_r - G_a - G_b) / (kB * T))
    return K

def calculate_rate(T, C_a, k=None, K=None, G_a=None, G_b=None, G_r=None, G_ts=None):
    kB = 1.380649e-23 / 4.3597447222071e-18
    h = 6.62607015e-34 / 4.3597447222071e-18
    T = numpy.array(T)
    C_a = numpy.array(C_a)
    if k is not None:
        logk = numpy.log(k)
    else:
        logk = numpy.log(calculate_k(G_r, G_ts, T))
    if K is not None:
        logKc = numpy.log(K * C_a)
    else:
        logKc = numpy.log(calculate_K(G_a, G_b, G_r, T) * C_a)
    rates = numpy.exp( logk + logKc - numpy.log(1.0 + numpy.exp(logKc)) )
    return rates

def rotate_vector(vector, axis, angle, degrees=True):
    unit = axis / numpy.linalg.norm(axis)
    parallel = numpy.inner(vector, unit) * unit
    perpend1 = vector - parallel
    perpend2 = numpy.cross(unit, perpend1)
    if degrees:
        rotated = parallel + perpend1 * numpy.cos(numpy.pi*angle/180.0) + perpend2 * numpy.sin(numpy.pi*angle/180.0)
    else:
        rotated = parallel + perpend1 * numpy.cos(angle) + perpend2 * numpy.sin(angle)
    return rotated

def make_zmatrix(atoms, coords):
    X_atoms = []
    X_coords = []
    H_atoms = []
    H_coords = []
    for X, coord in zip(atoms, coords):
        if X != 'H':
            X_atoms.append(X)
            X_coords.append(coord)
        else:
            H_atoms.append(X)
            H_coords.append(coord)
    n_X = len(X_atoms)
    molecule = Atoms(X_atoms + H_atoms, X_coords + H_coords)
    atoms = molecule.get_chemical_symbols()
    indices = []
    coords = []
    for i, X in enumerate(atoms):
        if X != 'H':
            if i >= 1:
                indices.append(i-1)
                coords.append(molecule.get_distance(i, i-1))
            if i >= 2:
                indices.append(i-2)
                coords.append(molecule.get_angle(i, i-1, i-2))
            if i >= 3:
                indices.append(i-3)
                coords.append(molecule.get_dihedral(i, i-1, i-2, i-3))
        else:
            if n_X >= 1:
                distances = molecule.get_distances(i, [j for j in range(n_X)])
                n = numpy.argmin(distances)
            else:
                n = None
            if n_X >= 1:
                indices.append(n)
                coords.append(molecule.get_distance(i, n))
            elif i >= 1:
                indices.append(i-1)
                coords.append(molecule.get_distance(i, i-1))
            if n_X >= 2:
                indices.append((n-1+n_X)%n_X)
                coords.append(molecule.get_angle(i, n, (n-1+n_X)%n_X))
            elif i >= 2:
                if n_X == 1:
                    indices.append(n_X)
                    coords.append(molecule.get_angle(i, n, n_X))
                else:
                    indices.append(i-2)
                    coords.append(molecule.get_angle(i, i-1, i-2))
            if n_X >= 3:
                indices.append((n-2+n_X)%n_X)
                coords.append(molecule.get_dihedral(i, n, (n-1+n_X)%n_X, (n-2+n_X)%n_X))
            elif i >= 3:
                if n_X == 2:
                    indices.append(n_X)
                    coords.append(molecule.get_dihedral(i, n, (n-1+n_X)%n_X, n_X))
                elif n_X == 1:
                    indices.append(n_X+1)
                    coords.append(molecule.get_dihedral(i, n, n_X, n_X+1))
                else:
                    indices.append(i-3)
                    coords.append(molecule.get_dihedral(i, i-1, i-2, i-3))
    return atoms, indices, coords


if __name__ == '__main__':

    from ase.io import read
    for file_path in ['tests/c0h2.xyz', 'tests/c1h4.xyz', 'tests/c2h6.xyz', 'tests/c3h8.xyz', 'tests/c6h12.xyz']:
        molecule = read(file_path, 0, 'xyz')
        atoms = molecule.get_chemical_symbols()
        coords = molecule.get_positions()
        atoms, indices, coords = make_zmatrix(atoms, coords)
        print(atoms)
        print([i+1 for i in indices])
        print(coords)
    for file_path in ['tests/A_0000.xyz']:
        molecule = read(file_path, 0, 'xyz')
        atoms = molecule.get_chemical_symbols()
        coords = molecule.get_positions()
        atoms, indices, coords = make_zmatrix(atoms[6:18], coords[6:18])
        print(atoms)
        print([i+1 for i in indices])
        print(coords)


import numpy

from ase import Atoms


def calc_k(T, G_r, G_t):
    kB = 1.380649e-23 / 4.3597447222071e-18
    h = 6.62607015e-34 / 4.3597447222071e-18
    T = numpy.array(T)
    G_r = numpy.array(G_r)
    G_t = numpy.array(G_t)
    return (kB * T / h) * numpy.exp(-(G_t - G_r) / (kB * T))

def calc_log_k(T, G_r, G_t):
    kB = 1.380649e-23 / 4.3597447222071e-18
    h = 6.62607015e-34 / 4.3597447222071e-18
    T = numpy.array(T)
    G_r = numpy.array(G_r)
    G_t = numpy.array(G_t)
    return numpy.log(kB * T / h) - (G_t - G_r) / (kB * T)

def calc_K(T, G_r, G_p):
    kB = 1.380649e-23 / 4.3597447222071e-18
    T = numpy.array(T)
    G_r = numpy.array(G_r)
    G_p = numpy.array(G_p)
    return numpy.exp(-(G_p - G_r) / (kB * T))

def calc_log_K(T, G_r, G_p):
    kB = 1.380649e-23 / 4.3597447222071e-18
    T = numpy.array(T)
    G_r = numpy.array(G_r)
    G_p = numpy.array(G_p)
    return -(G_p - G_r) / (kB * T)

def get_zmatrix_coords(coord, refs):
    atoms = (1 + len(refs)) * 'H'
    coords = [coord] + refs
    molecule = Atoms(atoms, coords)
    zmatrix = []
    if len(refs) >= 1:
        zmatrix.append(molecule.get_distance(0, 1))
    if len(refs) >= 2:
        zmatrix.append(molecule.get_angle(0, 1, 2))
    if len(refs) >= 3:
        zmatrix.append(molecule.get_dihedral(0, 1, 2, 3))
    return zmatrix

def get_zmatrix(atoms, coords):
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


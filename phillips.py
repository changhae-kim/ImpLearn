import numpy

from ase import Atoms
from ase.neighborlist import neighbor_list


class Phillips():

    def __init__(self, file_path, file_type, peripheral_oxygens,
            bond_cutoffs={
                ('Si', 'Si'): 2.0, ('O', 'O'): 2.0, ('Si', 'O'): 2.3, ('O', 'H'): 1.2,
                ('Cr', 'O'): 2.3, ('Cr', 'C'): 2.3, ('C', 'C'): 2.0, ('C', 'H'): 1.2
                },
            bond_lengths={('Cr', 'O'): 1.82, ('Cr', 'C'): 2.02, ('C', 'C'): 1.53, ('C', 'H'): 1.09},
            ethylene_bond_lengths={('Cr', 'C'): 2.5, ('C', 'C'): 1.34, ('C', 'H'): 1.09}
            ):

        self.bond_cutoffs = bond_cutoffs
        self.bond_lengths = bond_lengths
        self.ethylene_bond_lengths = ethylene_bond_lengths

        self.cluster = self.import_cluster(file_path, file_type)

        for n, i in enumerate(peripheral_oxygens):
            if i < 0:
                peripheral_oxygens[n] = len(self.slab.get_chemical_symbols()) + i

        self.axes = self.define_axes(self.cluster, peripheral_oxygens)
        self.chromium_cluster = self.add_chromium(self.cluster, peripheral_oxygens)
        self.L_ethyl_cluster = self.add_alkyl(self.chromium_cluster, 2, point_y=True, rotate_2=False)
        self.L_ethyl_R_ethylene_cluster = self.add_ethylene(self.L_ethyl_cluster, point_y=False)
        self.R_butyl_cluster = self.add_alkyl(self.chromium_cluster, 4, point_y=False, rotate_2=True)
        self.R_ethyl_cluster = self.add_alkyl(self.chromium_cluster, 2, point_y=False, rotate_2=False)
        self.R_ethyl_L_ethylene_cluster = self.add_ethylene(self.R_ethyl_cluster, point_y=True)
        self.L_butyl_cluster = self.add_alkyl(self.chromium_cluster, 4, point_y=True, rotate_2=True)

        return

    def import_cluster(self, file_path, file_type):
        from ase.io import read
        cluster = read(file_path, 0, file_type)
        return cluster

    def define_axes(self, cluster, peripheral_oxygens):

        atoms = self.cluster.get_chemical_symbols()
        coords = self.cluster.get_positions()
        bonds = neighbor_list('ij', self.cluster, self.bond_cutoffs)

        chasis_silicons = []
        for i in peripheral_oxygens:
            i_neighbors = bonds[1][bonds[0] == i]
            for j in i_neighbors:
                if atoms[j] == 'Si':
                    chasis_silicons.append(j)

        n, m = peripheral_oxygens
        p, q = chasis_silicons
        xaxis = coords[m] - coords[n]
        xaxis = xaxis / numpy.linalg.norm(xaxis)
        zaxis = coords[n] + coords[m] - coords[p] - coords[q]
        zaxis = zaxis - numpy.inner(zaxis, xaxis) * xaxis
        zaxis = zaxis / numpy.linalg.norm(zaxis)
        yaxis = numpy.cross(zaxis, xaxis)
        axes = numpy.array([xaxis, yaxis, zaxis])

        return axes

    def add_chromium(self, cluster, peripheral_oxygens):

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()
        bonds = neighbor_list('ij', cluster, self.bond_cutoffs)

        peripheral_hydrogens = []
        for i in peripheral_oxygens:
            i_neighbors = bonds[1][bonds[0] == i]
            for j in i_neighbors:
                if atoms[j] == 'H':
                    peripheral_hydrogens.append(j)

        n, m = peripheral_oxygens
        OO_distance = cluster.get_distance(n, m)
        if 0.5 * OO_distance < self.bond_lengths[('Cr', 'O')]:
            Cr_coord = 0.5 * (coords[n] + coords[m]) + self.axes[2] * ((self.bond_lengths[('Cr', 'O')])**2.0 - (0.5 * OO_distance)**2.0)**0.5
        else:
            Cr_coord = 0.5 * (coords[n] + coords[m])

        chromium_atoms = []
        chromium_coords = []
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if i in peripheral_hydrogens:
                if i == peripheral_hydrogens[0]:
                    chromium_atoms.append('Cr')
                    chromium_coords.append(Cr_coord)
            else:
                chromium_atoms.append(X)
                chromium_coords.append(coord)

        chromium_cluster = Atoms(chromium_atoms, chromium_coords)

        return chromium_cluster

    def add_alkyl(self, cluster, ncarbons, point_y=True, rotate_2=False):

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()
        bonds = neighbor_list('ij', cluster, self.bond_cutoffs)

        Cr_index = -1
        Cr_coord = []
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if X == 'Cr':
                Cr_index = i
                Cr_coord = coord

        if point_y:
            tilts = [
                    self.axes[2] * numpy.cos(numpy.pi*0.5*109.5/180.0) + self.axes[1] * numpy.sin(numpy.pi*0.5*109.5/180.0),
                    self.axes[2] * numpy.cos(numpy.pi*(1.5*109.5-180.0)/180.0) + self.axes[1] * numpy.sin(numpy.pi*(1.5*109.5-180.0)/180.0)
                    ]
        else:
            tilts = [
                    self.axes[2] * numpy.cos(numpy.pi*0.5*109.5/180.0) - self.axes[1] * numpy.sin(numpy.pi*0.5*109.5/180.0),
                    self.axes[2] * numpy.cos(numpy.pi*(1.5*109.5-180.0)/180.0) - self.axes[1] * numpy.sin(numpy.pi*(1.5*109.5-180.0)/180.0)
                    ]

        C_coords = [Cr_coord + tilts[0] * self.bond_lengths[('Cr', 'C')]]
        for i in range(1, ncarbons):
            C_coords.append(C_coords[-1] + tilts[i%len(tilts)] * self.bond_lengths[('C', 'C')])
        H_coords = []
        for i in range(0, ncarbons):
            if i == ncarbons-1:
                H_coords.append(C_coords[-1] + tilts[(i+1)%len(tilts)] * self.bond_lengths[('C', 'H')])
            H_coords.append(C_coords[i] + self.rotate_vector(tilts[(i+1)%len(tilts)], -tilts[(i+0)%len(tilts)], +120.0) * self.bond_lengths[('C', 'H')])
            H_coords.append(C_coords[i] + self.rotate_vector(tilts[(i+1)%len(tilts)], -tilts[(i+0)%len(tilts)], -120.0) * self.bond_lengths[('C', 'H')])

        if rotate_2:
            for i in range(2, ncarbons):
                C_coords[i] = C_coords[1] + self.rotate_vector(C_coords[i]-C_coords[1], -tilts[1], +120.0)
            for i in range(2, 2*ncarbons+1):
                H_coords[i] = C_coords[1] + self.rotate_vector(H_coords[i]-C_coords[1], -tilts[1], +120.0)

        n = len(atoms)+1
        alkyl_atoms = []
        alkyl_coords = []
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if X == 'C':
                n = i
                break
            else:
                alkyl_atoms.append(X)
                alkyl_coords.append(coord)
        for X, coord in zip(atoms[n:], coords[n:]):
            if X == 'C':
                alkyl_atoms.append('C')
                alkyl_coords.append(coord)
        for coord in C_coords:
            alkyl_atoms.append('C')
            alkyl_coords.append(coord)
        for X, coord in zip(atoms[n:], coords[n:]):
            if X == 'H':
                alkyl_atoms.append('H')
                alkyl_coords.append(coord)
        for coord in H_coords:
            alkyl_atoms.append('H')
            alkyl_coords.append(coord)

        alkyl_cluster = Atoms(alkyl_atoms, alkyl_coords)

        return alkyl_cluster

    def add_ethylene(self, cluster, point_y=False):

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()
        bonds = neighbor_list('ij', cluster, self.bond_cutoffs)

        Cr_index = -1
        Cr_coord = []
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if X == 'Cr':
                Cr_index = i
                Cr_coord = coord

        angle = numpy.arctan(0.5 * self.ethylene_bond_lengths[('C', 'C')] / self.ethylene_bond_lengths[('Cr', 'C')])
        if point_y:
            tilt0 = self.axes[2] * numpy.cos(numpy.pi*0.5*109.5/180.0) + self.axes[1] * numpy.sin(numpy.pi*0.5*109.5/180.0)
            tilt1 = tilt0 * numpy.cos(angle) + self.axes[0] * numpy.sin(angle)
            tilt2 = tilt0 * numpy.sin(angle) - self.axes[0] * numpy.cos(angle)
        else:
            tilt0 = self.axes[2] * numpy.cos(numpy.pi*0.5*109.5/180.0) - self.axes[1] * numpy.sin(numpy.pi*0.5*109.5/180.0)
            tilt1 = tilt0 * numpy.cos(angle) - self.axes[0] * numpy.sin(angle)
            tilt2 = tilt0 * numpy.sin(angle) + self.axes[0] * numpy.cos(angle)
        C1_coord = Cr_coord + tilt1 * self.ethylene_bond_lengths[('Cr', 'C')]
        C2_coord = C1_coord + tilt2 * self.ethylene_bond_lengths[('C', 'C')]
        C_coords = [C1_coord, C2_coord]
        H1_coord = C1_coord + self.rotate_vector(tilt2, tilt1, -120.0) * self.bond_lengths[('C', 'H')]
        H2_coord = C1_coord + self.rotate_vector(tilt2, tilt1, +120.0) * self.bond_lengths[('C', 'H')]
        H3_coord = C2_coord + self.rotate_vector(tilt2, tilt1, -60.0) * self.bond_lengths[('C', 'H')]
        H4_coord = C2_coord + self.rotate_vector(tilt2, tilt1, +60.0) * self.bond_lengths[('C', 'H')]
        H_coords = [H1_coord, H2_coord, H3_coord, H4_coord]

        n = len(atoms)+1
        ethylene_atoms = []
        ethylene_coords = []
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if X == 'C':
                n = i
                break
            else:
                ethylene_atoms.append(X)
                ethylene_coords.append(coord)
        for coord in C_coords:
            ethylene_atoms.append('C')
            ethylene_coords.append(coord)
        for X, coord in zip(atoms[n:], coords[n:]):
            if X == 'C':
                ethylene_atoms.append('C')
                ethylene_coords.append(coord)
        for coord in H_coords:
            ethylene_atoms.append('H')
            ethylene_coords.append(coord)
        for X, coord in zip(atoms[n:], coords[n:]):
            if X == 'H':
                ethylene_atoms.append('H')
                ethylene_coords.append(coord)

        ethylene_cluster = Atoms(ethylene_atoms, ethylene_coords)

        return ethylene_cluster

    def rotate_vector(self, vector, axis, angle, degrees=True):
        unit = axis / numpy.linalg.norm(axis)
        parallel = numpy.inner(vector, unit) * unit
        perpend1 = vector - parallel
        perpend2 = numpy.cross(unit, perpend1)
        if degrees:
            rotated = parallel + perpend1 * numpy.cos(numpy.pi*angle/180.0) + perpend2 * numpy.sin(numpy.pi*angle/180.0)
        else:
            rotated = parallel + perpend1 * numpy.cos(angle) + perpend2 * numpy.sin(angle)
        return rotated

    def export_clusters(self, file_path, file_type):
        from ase.io import write
        write(file_path.format('L_ethyl'), self.L_ethyl_cluster, file_type)
        write(file_path.format('L_ethyl_R_ethylene'), self.L_ethyl_R_ethylene_cluster, file_type)
        write(file_path.format('R_butyl'), self.R_butyl_cluster, file_type)
        write(file_path.format('R_ethyl'), self.R_ethyl_cluster, file_type)
        write(file_path.format('R_ethyl_L_ethylene'), self.R_ethyl_L_ethylene_cluster, file_type)
        write(file_path.format('L_butyl'), self.L_butyl_cluster, file_type)
        return


if __name__ == '__main__':

    clusters = Phillips('output_silanols/A_0000.xyz', 'xyz', [2, 3])
    print('--- MAIN ---')
    clusters.export_clusters('A_0000_{:s}.xyz', 'xyz')


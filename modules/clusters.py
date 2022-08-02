import numpy

import ase.io.read
import ase.io.write

from ase import Atoms
from ase.neighborlist import neighbor_list
from matplotlib import pyplot

from tools import rotate_vector


class Silanols():

    def __init__(self, file_path, file_type,
            bond_cutoffs = {('Si', 'Si'): 2.0, ('O', 'O'): 2.0, ('Si', 'O'): 2.3, ('O', 'H'): 1.2},
            vicinal_cutoff = 4.5,
            OH_bond_length = 0.96
            ):

        self.slab, self.bonds = self.load_slab(file_path, file_type, bond_cutoffs)
        self.atoms = self.slab.get_chemical_symbols()

        self.OH_groups = self.find_OH_groups()
        self.geminal_OH_pairs = self.find_geminal_OH_pairs()
        self.vicinal_OH_pairs = self.find_vicinal_OH_pairs(vicinal_cutoff)
        self.minimal_clusters = self.carve_minimal_clusters(OH_bond_length)

        return

    def load_slab(self, file_path, file_type, bond_cutoffs):
        slab = ase.io.read(file_path, 0, file_type)
        bonds = neighbor_list('ij', slab, bond_cutoffs)
        return slab, bonds

    def find_OH_groups(self, exclude_waters=True):
        OH_groups = []
        for i, X in enumerate(self.atoms):
            if X == 'H':
                H_neighbors = self.bonds[1][self.bonds[0] == i]
                if H_neighbors.shape[0] != 1:
                    print('find_OH_groups(): H {:d} is bonded to {:d} atoms'.format(i+1, H_neighbors.shape[0]))
                for j in H_neighbors:
                    if self.atoms[j] != 'O':
                        print('find_OH_groups(): H {:d} is bonded to {:s} {:d}'.format(i+1, self.atoms[j], j+1))
                for j in H_neighbors:
                    if self.atoms[j] == 'O':
                        O_neighbors = self.bonds[1][self.bonds[0] == j]
                        if O_neighbors.shape[0] != 2:
                            print('find_OH_groups(): O {:d} is bonded to {:d} atoms'.format(j+1, O_neighbors.shape[0]))
                        for k in O_neighbors:
                            if self.atoms[k] not in ['Si', 'H']:
                                print('find_OH_groups(): O {:d} is bonded to {:s} {:d}'.format(j+1, self.atoms[k], k+1))
                        if exclude_waters:
                            if 'Si' not in [self.atoms[i] for i in O_neighbors]:
                                continue
                            else:
                                OH_groups.append([i, j])
                        else:
                            if 'Si' not in [self.atoms[i] for i in O_neighbors]:
                                print('find_OH_groups(): O {:d} is not bonded to Si'.format(j+1))
                            OH_groups.append([i, j])
        return OH_groups

    def find_geminal_OH_pairs(self):
        geminal_OH_pairs = []
        for n, OH1_group in enumerate(self.OH_groups):
            for m, OH2_group in enumerate(self.OH_groups):
                if n > m:
                    O1_neighbors = self.bonds[1][self.bonds[0] == OH1_group[1]]
                    O2_neighbors = self.bonds[1][self.bonds[0] == OH2_group[1]]
                    if O1_neighbors.shape[0] != 2:
                        print('find_geminal_OH_pairs(): O {:d} is bonded to {:d} atoms'.format(OH1_group[1]+1, O1_neighbors.shape[0]))
                    for i in O1_neighbors:
                        if self.atoms[i] not in ['Si', 'H']:
                            print('find_geminal_OH_pairs(): O {:d} is bonded to {:s} {:d}'.format(OH1_group[1]+1, self.atoms[i], i+1))
                    if 'Si' not in [self.atoms[i] for i in O1_neighbors]:
                        print('find_geminal_OH_pairs(): O {:d} is not bonded to Si'.format(OH1_group[1]+1))
                    if O2_neighbors.shape[0] != 2:
                        print('find_geminal_OH_pairs(): O {:d} is bonded to {:d} atoms'.format(OH2_group[1]+1, O2_neighbors.shape[0]))
                    for i in O2_neighbors:
                        if self.atoms[i] not in ['Si', 'H']:
                            print('find_geminal_OH_pairs(): O {:d} is bonded to {:s} {:d}'.format(OH2_group[1]+1, self.atoms[i], i+1))
                    if 'Si' not in [self.atoms[i] for i in O2_neighbors]:
                        print('find_geminal_OH_pairs(): O {:d} is not bonded to Si'.format(OH2_group[1]+1))
                    for i in numpy.intersect1d(O1_neighbors, O2_neighbors):
                        if self.atoms[i] == 'Si':
                            geminal_OH_pairs.append([OH1_group, OH2_group])
        return geminal_OH_pairs

    def find_vicinal_OH_pairs(self, vicinal_cutoff, exclude_geminals=True):
        vicinal_OH_pairs = []
        for n, OH1_group in enumerate(self.OH_groups):
            for m, OH2_group in enumerate(self.OH_groups):
                if n > m:
                    if self.slab.get_distance(OH1_group[1], OH2_group[1], mic=True) < vicinal_cutoff:
                        if exclude_geminals:
                            geminal = False
                            for (OH3_group, OH4_group) in self.geminal_OH_pairs:
                                if OH1_group[1] in [OH3_group[1], OH4_group[1]] or OH2_group[1] in [OH3_group[1], OH4_group[1]]:
                                    geminal = True
                                    break
                            if geminal:
                                continue
                            else:
                                vicinal_OH_pairs.append([OH1_group, OH2_group])
                        else:
                            vicinal_OH_pairs.append([OH1_group, OH2_group])
        return vicinal_OH_pairs

    def carve_minimal_clusters(self, OH_bond_length, reorder_podals=True):

        minimal_clusters = []
        for (OH1_group, OH2_group) in self.vicinal_OH_pairs:

            atoms = []
            coords = []

            peripheral_hydrogens = [OH1_group[0], OH2_group[0]]
            atoms.append('H')
            coords.append(numpy.zeros((3,)))
            atoms.append('H')
            coords.append(coords[0] + self.slab.get_distance(OH1_group[0], OH2_group[0], mic=True, vector=True))

            peripheral_oxygens = [OH1_group[1], OH2_group[1]]
            atoms.append('O')
            coords.append(coords[0] + self.slab.get_distance(OH1_group[0], OH1_group[1], mic=True, vector=True))
            atoms.append('O')
            coords.append(coords[1] + self.slab.get_distance(OH2_group[0], OH2_group[1], mic=True, vector=True))

            chasis_silicons = []
            for n, i in enumerate(peripheral_oxygens):
                i_neighbors = self.bonds[1][self.bonds[0] == i]
                if i_neighbors.shape[0] != 2:
                    print('carve_minimal_clusters(): O {:d} is bonded to {:d} atoms'.format(i+1, i_neighbors.shape[0]))
                for j in i_neighbors:
                    if self.atoms[j] not in ['Si', 'H']:
                        print('carve_minimal_clusters(): O {:d} is bonded to {:s} {:d}'.format(i+1, self.atoms[j], j+1))
                if 'Si' not in [self.atoms[j] for j in i_neighbors]:
                    print('carve_minimal_clusters(): O {:d} is not bonded to Si'.format(i+1))
                for j in i_neighbors:
                    if self.atoms[j] == 'Si' and j not in chasis_silicons:
                        chasis_silicons.append(j)
                        atoms.append('Si')
                        coords.append(coords[len(peripheral_hydrogens) + n] + self.slab.get_distance(i, j, mic=True, vector=True))

            chasis_oxygens = []
            podal_oxygens = []
            for n, i in enumerate(chasis_silicons):
                i_neighbors = self.bonds[1][self.bonds[0] == i]
                if i_neighbors.shape[0] != 4:
                    print('carve_minimal_clusters(): Si {:d} is bonded to {:d} atoms'.format(i+1, i_neighbors.shape[0]))
                for j in i_neighbors:
                    if self.atoms[j] != 'O':
                        print('carve_minimal_clusters(): Si {:d} is bonded to {:s} {:d}'.format(i+1, self.atoms[j], j+1))
                for j in i_neighbors:
                    if self.atoms[j] == 'O' and j not in peripheral_oxygens + chasis_oxygens + podal_oxygens:
                        j_neighbors = self.bonds[1][self.bonds[0] == j]
                        m = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(chasis_oxygens)
                        if numpy.intersect1d(chasis_silicons, j_neighbors).shape[0] >= 2:
                            chasis_oxygens.append(j)
                            atoms.insert(m, 'O')
                            coords.insert(m, coords[m + n] + self.slab.get_distance(i, j, mic=True, vector=True))
                        else:
                            podal_oxygens.append(j)
                            atoms.append('O')
                            coords.append(coords[m + n] + self.slab.get_distance(i, j, mic=True, vector=True))

            if reorder_podals:
                n = len(peripheral_hydrogens)
                m = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(chasis_oxygens)
                p = m + len(chasis_silicons)
                q = m + len(chasis_silicons) + len(podal_oxygens)
                reordered = self.reorder_podal_oxygens(coords[p:q], coords[n+0], coords[n+1], coords[m+0], coords[m+1])
                old_podals = podal_oxygens
                old_coords = coords
                podal_oxygens = []
                coords = []
                for coord in old_coords[:p]:
                    coords.append(coord)
                for i in reordered:
                    podal_oxygens.append(old_podals[i])
                    coords.append(old_coords[p:q][i])
                for coord in old_coords[q:]:
                    coords.append(coord)

            podal_hydrogens = []
            for n, i in enumerate(podal_oxygens):
                i_neighbors = self.bonds[1][self.bonds[0] == i]
                if i_neighbors.shape[0] != 2:
                    print('carve_minimal_clusters(): O {:d} is bonded to {:d} atoms'.format(i+1, i_neighbors.shape[0]))
                for j in i_neighbors:
                    if self.atoms[j] not in ['Si', 'H']:
                        print('carve_minimal_clusters(): O {:d} is bonded to {:s} {:d}'.format(i+1, self.atoms[j], j+1))
                for j in i_neighbors:
                    if j not in peripheral_hydrogens + chasis_silicons:
                        m = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(chasis_oxygens) + len(chasis_silicons)
                        if self.atoms[j] == 'H':
                            podal_hydrogens.append(j)
                            atoms.append('H')
                            coords.append(coords[m + n] + self.slab.get_distance(i, j, mic=True, vector=True))
                        elif self.atoms[j] == 'Si':
                            podal_hydrogens.append(j)
                            atoms.append('H')
                            axis = self.slab.get_distance(i, j, mic=True, vector=True)
                            axis = axis / numpy.linalg.norm(axis)
                            coords.append(coords[m + n] + axis * OH_bond_length)

            cluster = Atoms(atoms, coords)
            minimal_clusters.append(cluster)

        return minimal_clusters

    def reorder_podal_oxygens(self, coords, O1_coord, O2_coord, Si1_coord, Si2_coord, max_iter=50):

        origin = 0.5 * (Si1_coord + Si2_coord)
        centered = coords - origin
        xaxis = Si2_coord - Si1_coord
        xaxis = xaxis / numpy.linalg.norm(xaxis)
        zaxis = 0.5 * (O1_coord + O2_coord - Si1_coord - Si2_coord)
        zaxis = zaxis / numpy.linalg.norm(zaxis)
        yaxis = numpy.cross(zaxis, xaxis)

        if len(coords) not in [2, 4, 6]:
            print('number of podal oxygens is {:d}'.format(len(coords)))

        centered1 = coords - Si1_coord
        centered2 = coords - Si2_coord
        reordered1 = []
        reordered2 = []
        for i, (coord1, coord2) in enumerate(zip(centered1, centered2)):
            if numpy.linalg.norm(coord1) < numpy.linalg.norm(coord2):
                reordered1.append(i)
            else:
                reordered2.append(i)

        zaxis1 = O1_coord - Si1_coord
        zaxis1 = zaxis1 / numpy.linalg.norm(zaxis1)
        status = -1
        for i in range(max_iter):
            if status == 0:
                break
            else:
                status = 0
                for i, _ in enumerate(reordered1[:-1]):
                    if numpy.dot(numpy.cross(zaxis1, centered1[reordered1[i]]), centered1[reordered1[i+1]]) < 0.0:
                        reordered1[i], reordered1[i+1] = reordered1[i+1], reordered1[i]
                        status = -1
                        break

        zaxis2 = O2_coord - Si2_coord
        zaxis2 = zaxis2 / numpy.linalg.norm(zaxis2)
        status = -1
        for i in range(max_iter):
            if status == 0:
                break
            else:
                status = 0
                for i, _ in enumerate(reordered2[:-1]):
                    if numpy.dot(numpy.cross(zaxis2, centered2[reordered2[i]]), centered2[reordered2[i+1]]) < 0.0:
                        reordered2[i], reordered2[i+1] = reordered2[i+1], reordered2[i]
                        status = -1
                        break

        if len(reordered1) > 2:
            if len(reordered2) > 2:
                nm = numpy.argmin([numpy.linalg.norm(centered[i]-centered[j]) for i in reordered1 for j in reordered2])
                n = nm//len(reordered2)
                m = nm%len(reordered2)
                reordered1 = [reordered1[(n+1+i)%len(reordered1)] for i, _ in enumerate(reordered1)]
                reordered2 = [reordered2[(m+i)%len(reordered2)] for i, _ in enumerate(reordered2)]
            else:
                n = numpy.argmin([numpy.linalg.norm(centered[i]-centered[0]) for i in reordered1])
                reordered1 = [reordered1[(n+1+i)%len(reordered1)] for i, _ in enumerate(reordered1)]
        elif len(reordered2) > 2:
            m = numpy.argmin([numpy.linalg.norm(centered[reordered1[-1]]-centered[j]) for j in reordered2])
            reordered2 = [reordered2[(m+i)%len(reordered2)] for i, _ in enumerate(reordered2)]
        reordered = reordered1 + reordered2

        return reordered

    def analyze_distances(self, file_path):
        distances = self.slab.get_all_distances(mic=True)
        elements = ['Si', 'O', 'H']
        for n, A in enumerate(elements):
            for m, B in enumerate(elements):
                if n <= m:
                    if n == m:
                        AB_distances = [distances[i, j] for i, X in enumerate(self.atoms) for j, Y in enumerate(self.atoms) if i < j and X == A and Y == B]
                    else:
                        AB_distances = [distances[i, j] for i, X in enumerate(self.atoms) for j, Y in enumerate(self.atoms) if X == A and Y == B]
                    pyplot.figure()
                    pyplot.hist(AB_distances, bins=[i * 0.1 for i in range(0, 51)])
                    pyplot.xlabel('{:s}-{:s} Distances (Å)'.format(A, B))
                    pyplot.ylabel('Frequency')
                    pyplot.tight_layout()
                    pyplot.savefig(file_path.format(A, B))
        return

    def save_clusters(self, file_path, file_type):
        for i, cluster in enumerate(self.minimal_clusters):
            ase.io.write(file_path.format(i), cluster, file_type)
        return


class Phillips():

    def __init__(self, file_path, file_type, peripheral_oxygens,
            alkyl_lengths=[4, 6],
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

        self.cluster = self.load_cluster(file_path, file_type)

        for n, i in enumerate(peripheral_oxygens):
            if i < 0:
                peripheral_oxygens[n] = len(self.slab.get_chemical_symbols()) + i

        self.axes = self.define_axes(self.cluster, peripheral_oxygens)
        self.chromium_cluster = self.add_chromium(self.cluster, peripheral_oxygens)
        self.L_butyl_cluster = self.add_alkyl(self.chromium_cluster, alkyl_lengths[0], point_y=True, rotate_2=False)
        self.L_butyl_R_ethylene_cluster = self.add_ethylene(self.L_butyl_cluster, point_y=False)
        self.R_hexyl_cluster = self.add_alkyl(self.chromium_cluster, alkyl_lengths[1], point_y=False, rotate_2=True)
        self.R_butyl_cluster = self.add_alkyl(self.chromium_cluster, alkyl_lengths[0], point_y=False, rotate_2=False)
        self.R_butyl_L_ethylene_cluster = self.add_ethylene(self.R_butyl_cluster, point_y=True)
        self.L_hexyl_cluster = self.add_alkyl(self.chromium_cluster, alkyl_lengths[1], point_y=True, rotate_2=True)

        return

    def load_cluster(self, file_path, file_type):
        cluster = ase.io.read(file_path, 0, file_type)
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
            H_coords.append(C_coords[i] + rotate_vector(tilts[(i+1)%len(tilts)], -tilts[(i+0)%len(tilts)], +120.0) * self.bond_lengths[('C', 'H')])
            H_coords.append(C_coords[i] + rotate_vector(tilts[(i+1)%len(tilts)], -tilts[(i+0)%len(tilts)], -120.0) * self.bond_lengths[('C', 'H')])

        if rotate_2:
            for i in range(2, ncarbons):
                C_coords[i] = C_coords[1] + rotate_vector(C_coords[i]-C_coords[1], -tilts[1], +120.0)
            for i in range(2, 2*ncarbons+1):
                H_coords[i] = C_coords[1] + rotate_vector(H_coords[i]-C_coords[1], -tilts[1], +120.0)

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
        H1_coord = C1_coord + rotate_vector(tilt2, tilt1, -120.0) * self.bond_lengths[('C', 'H')]
        H2_coord = C1_coord + rotate_vector(tilt2, tilt1, +120.0) * self.bond_lengths[('C', 'H')]
        H3_coord = C2_coord + rotate_vector(tilt2, tilt1, -60.0) * self.bond_lengths[('C', 'H')]
        H4_coord = C2_coord + rotate_vector(tilt2, tilt1, +60.0) * self.bond_lengths[('C', 'H')]
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

    def save_clusters(self, file_path, file_type):
        ase.io.write(file_path.format('L_butyl'), self.L_butyl_cluster, file_type)
        ase.io.write(file_path.format('L_butyl_R_ethylene'), self.L_butyl_R_ethylene_cluster, file_type)
        ase.io.write(file_path.format('R_hexyl'), self.R_hexyl_cluster, file_type)
        ase.io.write(file_path.format('R_butyl'), self.R_butyl_cluster, file_type)
        ase.io.write(file_path.format('R_butyl_L_ethylene'), self.R_butyl_L_ethylene_cluster, file_type)
        ase.io.write(file_path.format('L_hexyl'), self.L_hexyl_cluster, file_type)
        return


if __name__ == '__main__':

    clusters = Silanols('tests/A_117SiO2_35H2O', 'vasp')
    print('--- MAIN ---')
    print('clusters.atoms')
    print(clusters.atoms)
    print('clusters.bonds')
    print(clusters.bonds)
    print('clusters.OH_groups')
    print(len(clusters.OH_groups))
    print(clusters.OH_groups)
    print('clusters.geminal_OH_pairs')
    print(len(clusters.geminal_OH_pairs))
    print(clusters.geminal_OH_pairs)
    print('clusters.vicinal_OH_pairs')
    print(len(clusters.vicinal_OH_pairs))
    print(clusters.vicinal_OH_pairs)
    clusters.analyze_distances('A_d{:s}{:s}.png')
    clusters.save_clusters('A_{:04d}.xyz', 'xyz')

    clusters = Phillips('tests/A_0000.xyz', 'xyz', [2, 3])
    clusters.save_clusters('A_0000_{:s}.xyz', 'xyz')

